/*
 * Copyright 2026 The FlorisBoard Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dev.patrickgold.florisboard.ime.nlp.ru

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.ln

/**
 * Simple dictionary based autocorrect engine for Russian.
 *
 * The engine knows a large list of valid word forms ([known]) and a smaller pool of the most frequent words that
 * are used as correction candidates. A typed word that is neither known nor in the user dictionary is compared to
 * the candidate pool using a weighted Damerau-Levenshtein distance, where substituting a letter by a neighbouring
 * key of the JCUKEN layout is cheaper than an arbitrary substitution.
 */
object RuAutocorrect {
    private const val ASSET_PATH = "ru_autocorrect/ru_words.txt"
    private const val USER_WORDS_FILE = "ru_user_words.txt"
    private const val PREFS_NAME = "ru_autocorrect"
    private const val PREF_ENABLED = "enabled"

    private const val POOL_SIZE = 100_000
    private const val MAX_CANDIDATES = 6
    private const val SEARCH_CUTOFF = 2.0
    private const val AUTO_CUTOFF = 1.2
    private const val MIN_AUTO_COUNT = 50

    class Correction(
        /** The word as it was typed. */
        val original: String,
        /** Ranked corrections, already cased like [original]. The first one is applied automatically. */
        val candidates: List<String>,
    )

    @Volatile
    private var appContext: Context? = null

    @Volatile
    private var ready = false

    @Volatile
    var enabled = true
        private set

    private var known: Set<String> = emptySet()
    private var poolWords: Array<String> = emptyArray()
    private var poolNorm: Array<String> = emptyArray()
    private var poolCounts: IntArray = IntArray(0)
    private val userWords: MutableSet<String> = ConcurrentHashMap.newKeySet()
    private val userWordsLock = Any()

    // JCUKEN layout geometry used for the "neighbouring key" substitution cost.
    private val keyX = HashMap<Char, Double>()
    private val keyY = HashMap<Char, Int>()

    init {
        val rows = arrayOf("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")
        val offsets = doubleArrayOf(0.0, 0.25, 0.75)
        for ((y, row) in rows.withIndex()) {
            for ((x, c) in row.withIndex()) {
                keyX[c] = x + offsets[y]
                keyY[c] = y
            }
        }
    }

    /**
     * Starts loading the dictionaries on a background thread. Safe to call multiple times.
     */
    fun init(context: Context) {
        if (appContext != null) return
        val ctx = context.applicationContext ?: context
        appContext = ctx
        try {
            enabled = ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getBoolean(PREF_ENABLED, true)
        } catch (_: Exception) {
        }
        Thread({ load(ctx) }, "ru-autocorrect-load").apply {
            priority = Thread.MIN_PRIORITY
            start()
        }
    }

    private fun load(ctx: Context) {
        try {
            val knownSet = HashSet<String>(300_000)
            val words = ArrayList<String>(POOL_SIZE)
            val counts = ArrayList<Int>(POOL_SIZE)
            ctx.assets.open(ASSET_PATH).bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (line in lines) {
                    val sep = line.lastIndexOf(' ')
                    if (sep <= 0) continue
                    val word = line.substring(0, sep)
                    val count = line.substring(sep + 1).toIntOrNull() ?: continue
                    knownSet.add(norm(word))
                    if (words.size < POOL_SIZE) {
                        words.add(word)
                        counts.add(count)
                    }
                }
            }
            try {
                val file = File(ctx.filesDir, USER_WORDS_FILE)
                if (file.exists()) {
                    file.forEachLine { line ->
                        val w = line.trim()
                        if (w.isNotEmpty()) userWords.add(norm(w))
                    }
                }
            } catch (_: Exception) {
            }
            known = knownSet
            poolWords = words.toTypedArray()
            poolNorm = Array(words.size) { norm(words[it]) }
            poolCounts = counts.toIntArray()
            ready = true
        } catch (_: Throwable) {
            ready = false
        }
    }

    fun isReady(): Boolean = ready && enabled

    fun toggle() {
        enabled = !enabled
        val ctx = appContext ?: return
        try {
            ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, enabled).apply()
        } catch (_: Exception) {
        }
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(
                ctx,
                if (enabled) "Автозамена включена" else "Автозамена выключена",
                Toast.LENGTH_SHORT,
            ).show()
        }
    }

    fun showToast(text: String) {
        val ctx = appContext ?: return
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(ctx, text, Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Adds [word] to the user dictionary, so it is never corrected again.
     */
    fun learn(word: String) {
        val key = norm(word)
        if (key.isEmpty() || !userWords.add(key)) return
        val ctx = appContext ?: return
        Thread {
            try {
                synchronized(userWordsLock) {
                    File(ctx.filesDir, USER_WORDS_FILE).appendText(key + "\n", Charsets.UTF_8)
                }
            } catch (_: Exception) {
            }
        }.start()
    }

    private fun norm(s: String): String = s.lowercase().replace('ё', 'е')

    private fun isKnown(word: String): Boolean {
        val key = norm(word)
        return known.contains(key) || userWords.contains(key)
    }

    /**
     * Returns a correction for [typed] or null, if the word is fine or no confident correction exists.
     */
    fun correct(typed: String): Correction? {
        if (!isReady()) return null
        if (typed.length < 3 || typed.length > 24) return null
        if (typed.all { it.isUpperCase() }) return null
        if (isKnown(typed)) return null

        val target = norm(typed)
        val len = target.length
        val hits = ArrayList<Hit>()
        val bufSize = len + 4
        val rowA = DoubleArray(bufSize)
        val rowB = DoubleArray(bufSize)
        val rowC = DoubleArray(bufSize)
        val words = poolWords
        val norms = poolNorm
        val counts = poolCounts
        for (i in norms.indices) {
            val cand = norms[i]
            val dl = cand.length - len
            if (dl < -2 || dl > 2) continue
            val d = distance(target, cand, SEARCH_CUTOFF, rowA, rowB, rowC)
            if (d <= SEARCH_CUTOFF) {
                hits.add(Hit(words[i], d, counts[i], ln(counts[i].toDouble() + 1.0) - 4.0 * d))
            }
        }
        if (hits.isEmpty()) return null
        hits.sortByDescending { it.score }

        val limit = if (len == 3) 1.0 else AUTO_CUTOFF
        val auto = hits.firstOrNull {
            it.cost <= limit && it.count >= MIN_AUTO_COUNT && (len > 3 || it.count >= 3000)
        } ?: return null

        val ordered = ArrayList<Hit>()
        ordered.add(auto)
        for (h in hits) {
            if (ordered.size >= MAX_CANDIDATES) break
            if (h !== auto) ordered.add(h)
        }
        val capitalize = typed.first().isUpperCase()
        val candidates = ordered.map { applyCase(it.word, capitalize) }.distinct()
        return Correction(typed, candidates)
    }

    private fun applyCase(word: String, capitalize: Boolean): String {
        return if (capitalize) word.replaceFirstChar { it.uppercaseChar() } else word
    }

    private class Hit(val word: String, val cost: Double, val count: Int, val score: Double)

    private fun substitutionCost(a: Char, b: Char): Double {
        if (a == b) return 0.0
        val ax = keyX[a]
        val bx = keyX[b]
        val ay = keyY[a]
        val by = keyY[b]
        if (ax != null && bx != null && ay != null && by != null) {
            if (abs(ay - by) <= 1 && abs(ax - bx) <= 1.0) return 0.6
        }
        return 1.0
    }

    /**
     * Weighted optimal string alignment distance. Returns a value greater than [cutoff] as soon as it is clear
     * that the distance exceeds it.
     */
    private fun distance(
        a: String,
        b: String,
        cutoff: Double,
        buf0: DoubleArray,
        buf1: DoubleArray,
        buf2: DoubleArray,
    ): Double {
        val n = a.length
        val m = b.length
        if (abs(n - m) > cutoff) return cutoff + 1.0
        var p2 = buf0
        var p1 = buf1
        var c = buf2
        for (j in 0..m) p1[j] = j.toDouble()
        for (i in 1..n) {
            c[0] = i.toDouble()
            var rowMin = c[0]
            val ca = a[i - 1]
            for (j in 1..m) {
                val cb = b[j - 1]
                var v = p1[j - 1] + substitutionCost(ca, cb)
                val del = p1[j] + 1.0
                if (del < v) v = del
                val ins = c[j - 1] + 1.0
                if (ins < v) v = ins
                if (i > 1 && j > 1 && ca == b[j - 2] && a[i - 2] == cb) {
                    val tr = p2[j - 2] + 0.9
                    if (tr < v) v = tr
                }
                c[j] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > cutoff) return cutoff + 1.0
            val tmp = p2
            p2 = p1
            p1 = c
            c = tmp
        }
        return p1[m]
    }
}
