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

package dev.patrickgold.florisboard.ime.nlp.autocorrect

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.compose.ui.graphics.vector.ImageVector
import dev.patrickgold.florisboard.ime.nlp.SuggestionCandidate
import dev.patrickgold.florisboard.ime.nlp.SuggestionProvider
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.ln

/**
 * Simple dictionary based autocorrect for Russian and English.
 *
 * Every language knows a large list of valid word forms and a smaller pool of the most frequent words that are used
 * as correction candidates. A typed word that is neither known nor in the user dictionary is compared to the pool
 * using a weighted Damerau-Levenshtein distance, where substituting a letter by a neighbouring key of the keyboard
 * is cheaper than an arbitrary substitution.
 */
object Autocorrect {
    private const val PREFS_NAME = "ru_autocorrect"
    private const val PREF_ENABLED = "enabled"

    private const val POOL_SIZE = 100_000
    private const val MAX_CANDIDATES = 6
    private const val SEARCH_CUTOFF = 2.0
    private const val AUTO_CUTOFF = 1.0
    private const val MIN_AUTO_COUNT = 50

    // A known word that is rare (below DOMINATED_MAX_COUNT) is treated as a typo, if a close word
    // is at least DOMINANCE_RATIO times more frequent. The dictionaries are built from subtitles and
    // contain a lot of typos with very low counts.
    private const val DOMINATED_MAX_COUNT = 500
    private const val DOMINANCE_RATIO = 60
    private const val MIN_DOMINANT_COUNT = 2000

    class Correction(
        /** The word as it was typed. */
        val original: String,
        /** Ranked corrections, already cased like [original]. The first one is applied automatically. */
        val candidates: List<String>,
    )

    private class LangConfig(
        val code: String,
        val assetPath: String,
        val userFile: String,
        val keyRows: Array<String>,
        /** Matches a whole word that can be autocorrected. */
        val wordRegex: Regex,
        /** Matches if the text ends with a letter of this language. */
        val endRegex: Regex,
        /** Names, brands etc. that are not (or rarely) in the frequency dictionary. */
        val extraWords: List<String>,
        /** Typed (lowercase) -> replacement. Applied regardless of the dictionary, e.g. dont -> don't. */
        val forced: Map<String, String>,
        /** Added to the distance if a candidate of a different length starts with another letter. */
        val firstLetterPenalty: Double,
    )

    private val configs: Map<String, LangConfig> = linkedMapOf(
        "ru" to LangConfig(
            code = "ru",
            assetPath = "ru_autocorrect/ru_words.txt",
            userFile = "ru_user_words.txt",
            keyRows = arrayOf("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"),
            wordRegex = Regex("[А-Яа-яЁё]+"),
            endRegex = Regex("[А-Яа-яЁё]$"),
            extraWords = listOf(
                "телеграм", "телеграма", "ватсап", "вотсап", "вайбер", "инстаграм", "ютуб", "гугл", "яндекс",
                "вконтакте", "тикток", "сбербанк", "сбер", "озон", "авито", "алиса", "вайлдберриз", "андроид",
                "айфон", "вайфай", "ковдор", "ковдора", "ковдору", "ковдором", "ковдоре", "ковдорский",
                "ковдорская", "ковдорского", "ковдорском", "мурманск", "мурманска", "мурманске", "мурманская",
                "мурманской", "мончегорск", "апатиты", "оленегорск", "кировск", "ловозеро", "полярные", "зори",
                "гок", "тэц", "ппр", "лот",
            ),
            forced = emptyMap(),
            firstLetterPenalty = 0.0,
        ),
        "en" to LangConfig(
            code = "en",
            assetPath = "en_autocorrect/en_words.txt",
            userFile = "en_user_words.txt",
            keyRows = arrayOf("qwertyuiop", "asdfghjkl", "zxcvbnm"),
            wordRegex = Regex("[A-Za-z][A-Za-z']*"),
            endRegex = Regex("[A-Za-z]$"),
            extraWords = listOf(
                "whatsapp", "telegram", "youtube", "iphone", "ipad", "android", "google", "facebook", "instagram",
                "tiktok", "twitter", "netflix", "amazon", "spotify", "discord", "snapchat", "linkedin", "wifi",
                "bluetooth", "email", "emails", "online", "app", "apps", "ok", "okay", "chatgpt", "gmail", "paypal",
                "bitcoin", "usb", "pdf", "sms", "gps", "btw", "thx", "thnx", "pls", "plz", "lol", "omg",
                "idk", "imo", "imho", "fyi", "asap", "tbh", "brb", "np", "ty", "ttyl", "rofl", "lmao",
                "wtf", "afaik", "diy", "eta", "faq", "dm", "pm",
            ),
            forced = mapOf(
                "i" to "I", "i'm" to "I'm", "i'll" to "I'll", "i've" to "I've", "i'd" to "I'd",
                "im" to "I'm", "ive" to "I've",
                "dont" to "don't", "cant" to "can't", "wont" to "won't", "doesnt" to "doesn't",
                "didnt" to "didn't", "isnt" to "isn't", "arent" to "aren't", "wasnt" to "wasn't",
                "werent" to "weren't", "hasnt" to "hasn't", "havent" to "haven't", "hadnt" to "hadn't",
                "couldnt" to "couldn't", "wouldnt" to "wouldn't", "shouldnt" to "shouldn't",
                "youre" to "you're", "theyre" to "they're", "thats" to "that's", "whats" to "what's",
                "hes" to "he's", "shes" to "she's", "weve" to "we've", "theyve" to "they've",
                "youve" to "you've", "youll" to "you'll", "theyll" to "they'll", "theres" to "there's",
                "heres" to "here's", "whos" to "who's",
                // very common misspellings the distance search can not resolve reliably
                "wich" to "which", "whcih" to "which", "tommorow" to "tomorrow", "tomorow" to "tomorrow",
                "tommorrow" to "tomorrow", "definatly" to "definitely", "definately" to "definitely",
                "definitly" to "definitely", "occured" to "occurred", "occurence" to "occurrence",
                "restraunt" to "restaurant", "ocassion" to "occasion", "unforgetable" to "unforgettable",
                "alot" to "a lot", "wierd" to "weird", "recieve" to "receive", "adress" to "address",
                "seperate" to "separate", "untill" to "until", "thier" to "their", "becuase" to "because",
                "beleive" to "believe", "freind" to "friend", "calender" to "calendar", "truely" to "truly",
                "teh" to "the", "taht" to "that", "adn" to "and", "jsut" to "just",
            ),
            firstLetterPenalty = 0.4,
        ),
    )

    @Volatile
    private var appContext: Context? = null

    @Volatile
    var enabled = true
        private set

    private val correctors = ConcurrentHashMap<String, Corrector>()

    fun supports(lang: String): Boolean = configs.containsKey(lang)

    fun wordRegex(lang: String): Regex? = configs[lang]?.wordRegex

    fun endRegex(lang: String): Regex? = configs[lang]?.endRegex

    fun isReady(lang: String): Boolean = enabled && correctors[lang]?.ready == true

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
        Thread({
            for (cfg in configs.values) {
                val corrector = Corrector(cfg)
                correctors[cfg.code] = corrector
                corrector.load(ctx)
            }
        }, "autocorrect-load").apply {
            priority = Thread.MIN_PRIORITY
            start()
        }
    }

    fun toggle() {
        enabled = !enabled
        val ctx = appContext ?: return
        try {
            ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().putBoolean(PREF_ENABLED, enabled).apply()
        } catch (_: Exception) {
        }
        showToast(if (enabled) "Автозамена включена" else "Автозамена выключена")
    }

    fun showToast(text: String) {
        val ctx = appContext ?: return
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(ctx, text, Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Adds [word] to the user dictionary of [lang], so it is never corrected again.
     */
    fun learn(lang: String, word: String) {
        val ctx = appContext ?: return
        correctors[lang]?.learn(ctx, word)
    }

    /**
     * Returns a correction for [typed] or null, if the word is fine or no confident correction exists.
     */
    fun correct(lang: String, typed: String): Correction? {
        if (!enabled) return null
        return correctors[lang]?.correct(typed)
    }

    private class Corrector(private val cfg: LangConfig) {
        @Volatile
        var ready = false

        private var knownCounts: Map<String, Int> = emptyMap()
        private var poolWords: Array<String> = emptyArray()
        private var poolNorm: Array<String> = emptyArray()
        private var poolCounts: IntArray = IntArray(0)
        private val userWords: MutableSet<String> = ConcurrentHashMap.newKeySet()
        private val userWordsLock = Any()

        // Keyboard geometry used for the "neighbouring key" substitution cost.
        private val keyX = HashMap<Char, Double>()
        private val keyY = HashMap<Char, Int>()

        init {
            val offsets = doubleArrayOf(0.0, 0.25, 0.75)
            for ((y, row) in cfg.keyRows.withIndex()) {
                for ((x, c) in row.withIndex()) {
                    keyX[c] = x + offsets[y]
                    keyY[c] = y
                }
            }
        }

        private fun norm(s: String): String {
            val lower = s.lowercase()
            return if (cfg.code == "ru") lower.replace('ё', 'е') else lower
        }

        fun load(ctx: Context) {
            try {
                val knownMap = HashMap<String, Int>(300_000)
                val words = ArrayList<String>(POOL_SIZE)
                val counts = ArrayList<Int>(POOL_SIZE)
                ctx.assets.open(cfg.assetPath).bufferedReader(Charsets.UTF_8).useLines { lines ->
                    for (line in lines) {
                        val sep = line.lastIndexOf(' ')
                        if (sep <= 0) continue
                        val word = line.substring(0, sep)
                        val count = line.substring(sep + 1).toIntOrNull() ?: continue
                        val nk = norm(word)
                        val prevCount = knownMap[nk]
                        if (prevCount == null || count > prevCount) knownMap[nk] = count
                        if (words.size < POOL_SIZE) {
                            words.add(word)
                            counts.add(count)
                        }
                    }
                }
                for (extra in cfg.extraWords) knownMap[norm(extra)] = 5000
                for (replacement in cfg.forced.values) knownMap[norm(replacement)] = 1_000_000
                try {
                    val file = File(ctx.filesDir, cfg.userFile)
                    if (file.exists()) {
                        file.forEachLine { line ->
                            val w = line.trim()
                            if (w.isNotEmpty()) userWords.add(norm(w))
                        }
                    }
                } catch (_: Exception) {
                }
                knownCounts = knownMap
                poolWords = words.toTypedArray()
                poolNorm = Array(words.size) { norm(words[it]) }
                poolCounts = counts.toIntArray()
                ready = true
            } catch (_: Throwable) {
                ready = false
            }
        }

        fun learn(ctx: Context, word: String) {
            val key = norm(word)
            if (key.isEmpty() || !userWords.add(key)) return
            Thread {
                try {
                    synchronized(userWordsLock) {
                        File(ctx.filesDir, cfg.userFile).appendText(key + "\n", Charsets.UTF_8)
                    }
                } catch (_: Exception) {
                }
            }.start()
        }

        fun correct(typed: String): Correction? {
            if (!ready) return null
            if (typed.isEmpty() || typed.length > 24) return null
            if (typed.length >= 2 && typed.all { it.isUpperCase() }) return null
            val target = norm(typed)
            if (userWords.contains(target)) return null

            val forcedWord = cfg.forced[target]
            if (forcedWord != null) {
                val out = if (typed.first().isUpperCase()) {
                    forcedWord.replaceFirstChar { it.uppercaseChar() }
                } else {
                    forcedWord
                }
                return if (out == typed) null else Correction(typed, listOf(out))
            }

            if (typed.length < 3) return null
            if (typed.contains('\'')) return null

            val ownCount: Int? = knownCounts[target]
            // Frequent known words are always fine.
            if (ownCount != null && ownCount >= DOMINATED_MAX_COUNT) return null

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
                var d = distance(target, cand, SEARCH_CUTOFF, rowA, rowB, rowC)
                if (d <= 0.0 || d > SEARCH_CUTOFF) continue
                if (cfg.firstLetterPenalty > 0.0 && cand.length != len && cand[0] != target[0]) {
                    d += cfg.firstLetterPenalty
                    if (d > SEARCH_CUTOFF) continue
                }
                hits.add(Hit(words[i], d, counts[i], ln(counts[i].toDouble() + 1.0) - 4.0 * d))
            }
            if (hits.isEmpty()) return null
            hits.sortByDescending { it.score }

            val limit = if (len == 3) 1.0 else AUTO_CUTOFF
            val auto = if (ownCount == null) {
                hits.firstOrNull {
                    it.cost <= limit && it.count >= MIN_AUTO_COUNT && (len > 3 || it.count >= 3000)
                }
            } else {
                // Known but rare word: only correct it, if a very close word is much more frequent.
                hits.firstOrNull {
                    it.cost <= 1.0 && it.count >= MIN_DOMINANT_COUNT &&
                        it.count.toLong() >= DOMINANCE_RATIO.toLong() * ownCount
                }
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

        private fun gap(c: Char): Double = if (c == '\'') 0.5 else 1.0

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
            p1[0] = 0.0
            for (j in 1..m) p1[j] = p1[j - 1] + gap(b[j - 1])
            var col0 = 0.0
            for (i in 1..n) {
                val ca = a[i - 1]
                col0 += gap(ca)
                c[0] = col0
                var rowMin = c[0]
                for (j in 1..m) {
                    val cb = b[j - 1]
                    var v = p1[j - 1] + substitutionCost(ca, cb)
                    val del = p1[j] + gap(ca)
                    if (del < v) v = del
                    val ins = c[j - 1] + gap(cb)
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
}

/**
 * One of (up to) three words shown in the suggestion strip after an autocorrection:
 * slot 0 = the typed word, slot 1 = the applied correction, slot 2 = an alternative correction.
 */
data class CorrectionCandidate(
    val slot: Int,
    override val text: CharSequence,
    val selected: Boolean,
) : SuggestionCandidate {
    override val secondaryText: CharSequence? = if (selected) "●" else null
    override val confidence: Double = 1.0
    override val isEligibleForAutoCommit: Boolean = false
    override val isEligibleForUserRemoval: Boolean = false
    override val icon: ImageVector? = null
    override val sourceProvider: SuggestionProvider? = null
}
