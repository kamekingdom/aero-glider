import string

def generate_combinations():
    # 2文字の全通りの配列bを生成
    alphabet = string.ascii_lowercase
    combinations = [c1 + c2 for c1 in alphabet for c2 in alphabet]
    return combinations

def remove_substrings(word_list):
    combinations = generate_combinations()

    # 配列bから存在する２文字の並びを削除
    for word in word_list:
        for substring in combinations:
            if substring in word:
                combinations.remove(substring)
    return combinations


# 高頻度の単語リストの作成
frequent = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "born", "various","am","are","was","were", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "hacker",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "hi",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "were", "has", "had", "are", "were", "will", "be",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "spring", "summer", "autumn", "fall", "winter",
    "apple", "banana", "carrot", "dog", "elephant", "fish", "guitar", "horse", "jungle",
    "kangaroo", "lion", "monkey", "night", "ocean", "piano", "queen", "rainbow", "sun", "tiger",
    "umbrella", "volcano", "whale", "xylophone", "yoga", "zebra", "airport", "butterfly", "camera", "dolphin",
    "elephant", "forest", "garden", "hiking", "island", "jacket", "kite", "laptop", "mountain", "notebook",
    "oasis", "pencil", "queen", "river", "sailboat", "tropical", "unicorn", "violin", "waterfall", "xylophone",
    "yacht", "zeppelin", "adventure", "basketball", "cactus", "desert", "elephant", "festival", "guitar", "happiness",
    "island", "jungle", "koala", "lighthouse", "moon", "nature", "ocean", "paradise", "quartz", "rainforest",
    "sunrise", "treasure", "underwater", "volcano", "watermelon", "xylophone", "yoga", "zebra", "ancient", "beauty",
    "castle", "daisy", "elephant", "forest", "garden", "harmony", "inspiration", "journey", "kiss", "laughter",
    "meadow", "nightingale", "oasis", "peace", "quiet", "reflection", "serenade", "tranquil", "universe", "victory",
    "whisper", "xylophone", "youthful", "zeal", "adventure", "balloon", "captivating", "dream", "enchanted", "fantasy",
    "giggles", "harmony", "imagination", "joyful", "kindness", "lullaby", "magic", "nightfall", "optimism", "playful",
    "red", "green", "yellow", "blue", "purple", "orange", "kame", "turtle", "thank", "toilet", "bathroom", "living", "kitchen", "here", "there", "pleasure"
]
verbs = [
    "be", "have", "do", "say", "go", "can", "get", "would", "make", "know",
    "will", "think", "take", "see", "come", "could", "want", "look", "use", "find",
    "give", "tell", "work", "may", "should", "call", "try", "ask", "need", "feel",
    "become", "leave", "put", "mean", "keep", "let", "begin", "seem", "help", "talk",
    "turn", "start", "show", "hear", "play", "run", "move", "like", "live", "believe",
    "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet",
    "include", "continue", "set", "learn", "change", "lead", "understand", "watch", "follow", "stop",
    "create", "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win",
    "offer", "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send",
    "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain", "suggest", "raise",
    "pass", "sell", "require", "report", "decide", "pull", "return", "break", "offer", "develop",
    "receive", "agree", "support", "continue", "improve", "join", "explain", "pick", "wear", "win",
    "listen", "operate", "write", "choose", "fight", "push", "thank", "throw", "act", "recognize",
    "realize", "manage", "raise", "visit", "rest", "teach", "finish", "suggest", "fly", "eat",
    "step", "wake", "hope", "kill", "sing", "create", "acknowledge", "spread", "play", "score",
    "shoot", "drink", "wear", "break", "shout", "test", "introduce", "enjoy", "invest", "apply",
    "manage", "shop", "provide", "win", "kill", "offer", "charge", "communicate", "produce", "forget",
    "exist", "stare", "rule", "explain", "wash", "park", "bury", "hide", "survive", "work",
    "respect", "present", "disappear", "suggest", "dance", "complain", "admit", "retire", "gather", "organize",
    "pray", "grin", "star", "destroy", "like", "devour", "beat", "suppose", "beat", "fit",
    "recognize", "consist", "construct", "hire", "form", "cook", "yell", "indicate", "organize", "enter",
    "ignore", "treat", "admire", "apologize", "inspect", "influence", "shut", "claim", "read", "protect",
    "publish", "suck", "insist", "march", "travel", "dare", "invent", "deliver", "paint", "pick",
    "adapt", "jump", "bother", "consult", "mind", "print", "rule", "manage", "dance", "ring",
    "drink", "state", "hate", "guarantee", "brush", "hire", "secure", "release", "love", "inquire",
    "observe", "qualify", "await", "deny", "knock", "ring", "clean", "resist", "reject", "greet",
    "beg", "blink", "register", "record", "deserve", "anticipate", "reside", "cancel", "accompany", "grasp",
    "compete", "reserve", "encourage", "pat", "ease", "hunt", "wonder", "nod", "strain", "flash",
    "pour", "upgrade", "relate", "amuse", "recruit", "occupy", "request", "supply", "fancy", "wander",
    "update", "grab", "vanish", "taste", "proclaim", "undergo", "borrow", "drift", "scream", "desire",
    "acquire", "admire", "blast", "unite", "reverse", "penetrate", "wipe", "fold", "plant", "declare",
    "desert", "adore", "characterize", "interrupt", "tolerate", "tap", "rescue", "summon", "plunge", "rest",
    "crawl", "float", "scan", "tap", "insert", "melt", "sail", "marry", "warn", "interpret",
    "blast", "evolve", "found", "train", "clean", "penetrate", "hunt", "appreciate", "trap", "retreat",
    "shatter", "select", "consult", "endorse", "reside", "strip", "provoke", "assemble", "cherish", "relax",
    "stumble", "clean", "pursue", "exit", "dictate", "forgive", "conceal", "pause", "mutter", "trap",
    "greet", "deprive", "bark", "scare", "cite", "unload", "substitute", "decipher", "rain", "lurk",
    "ascend", "sip", "mend", "bounce", "formulate", "rejoice", "recite", "rattle", "smoke", "float",
    "circulate", "confess", "grumble", "transform", "approve", "fade", "mug", "seize", "calculate", "enforce",
    "pronounce", "recruit", "preside", "crush", "crave", "attract", "terminate", "fasten", "steer", "resemble",
    "nurse", "peel", "penetrate", "boast", "crawl", "whisper", "inspire", "interfere", "repair", "loathe",
    "float", "sweep", "hammer", "reign", "shine", "fold", "fling", "float", "warn", "attract",
    "donate", "spark", "crawl", "allege", "march", "attract", "reign", "sigh", "excel", "contribute",
    "sweep", "shiver", "flip", "load", "pump", "deem", "plant", "tap", "trumpet", "gasp",
    "invoke", "sift", "aid", "pour", "insist", "yearn", "filter", "plunge", "upset", "consult",
    "trumpet", "squeal", "plunge", "wail", "forbid", "tailor", "catalog", "tickle", "crouch", "shuffle",
    "gobble", "filter", "tweak", "catalog", "polish", "yield", "contemplate", "hover", "meander", "expose",
    "lure", "drain", "crouch", "slam", "hurl", "upset", "yank", "blast", "yawn", "dedicate",
    "prop", "lick", "paddle", "twist", "hover", "applaud", "groan", "launch", "vow", "launch",
    "canoe", "drip", "blaze", "recite", "sip", "trick", "veer", "impart", "surge", "pause",
    "hop", "dab", "fidget", "exchange", "drip", "tweak", "plead", "exclaim", "douse", "curb",
    "volunteer", "unleash", "latch", "propel", "chatter", "forge", "muffle", "blurt", "surpass", "incite",
    "dazzle", "erupt", "unveil", "scoff", "gaze", "drown", "stall", "forge", "bruise", "taunt",
    "praise", "untangle", "undermine", "graze", "administer", "hammer", "clasp", "exert", "stagger", "detain",
    "conjure", "recount", "affirm", "stow", "confer", "maneuver", "jog", "latch", "sneak", "inspect",
    "prod", "harvest", "orchestrate", "counter", "hobble", "acquire", "safeguard", "tackle", "displace", "survey",
    "trample", "pinch", "convey", "redeem", "commute", "stash", "hem", "intercept", "navigate", "betray",
    "vault", "tame", "intensify", "saddle", "snatch", "snag", "access", "haunt", "spew", "deter",
    "impede", "gallop", "thump", "gasp", "mop", "exclaim", "weave", "prick", "latch", "flap",
    "multiply", "berate", "bleed", "gag", "ransack", "waft", "shiver", "grin", "gush", "despise",
    "gloat", "calibrate", "backfire", "sway", "audit", "summons", "caress", "obstruct", "dabble", "sprint",
    "squirm", "detest", "ponder", "latch", "plod", "pummel", "trudge", "rumble", "shimmer", "wield"
]
noun = [
    "time", "year", "people", "way", "day", "man", "government", "life", "company", "child",
    "group", "problem", "fact", "hand", "part", "place", "case", "week", "system", "program",
    "question", "work", "government", "number", "night", "point", "home", "water", "room", 
    "mother", "area", "money", "story", "fact", "month", "lot", "right", "study", "book",
    "eye", "job", "word", "business", "power", "country", "house", "friend", "father", "city",
    "school", "game", "line", "end", "member", "law", "car", "family", "state", "person",
    "student", "city", "problem", "team", "minute", "idea", "food", "information", "air",
    "fact", "force", "service", "trade", "history", "group", "university", "control", "death",
    "energy", "value", "effect", "investment", "rate", "team", "mind", "education", "community",
    "support", "sense", "nature", "goal", "production", "reason", "research", "project", "idea",
    "plan", "leader", "voice", "price", "industry", "technology", "value", "decision", "size",
    "change", "internet", "picture", "morning", "lawyer", "success", "message", "education",
    "window", "property", "daughter", "world", "relationship", "lady", "teacher", "phone",
    "health", "art", "war", "party", "music", "month", "method", "position", "data", "group",
    "car", "camera", "moment", "truth", "quality", "skill", "strategy", "way", "role", "idea",
    "heart", "news", "attention", "action", "street", "image", "baby", "purpose", "discussion",
    "model", "land", "town", "room", "name", "friend", "resource", "training", "college",
    "unit", "minute", "decision", "report", "girl", "phone", "application", "concept", "boy",
    "hour", "freedom", "stage", "relation", "nature", "hospital", "value", "article", "future",
    "traffic", "population", "software", "chance", "guy", "vote", "marriage", "role", "knowledge",
    "quality", "event", "mind", "river", "mind", "book", "team", "union", "style", "player",
    "pollution", "weapon", "service", "sister", "photo", "film", "flight", "experience",
    "strength", "truth", "race", "officer", "writer", "faith", "song", "employment", "guard",
    "detail", "wind", "figure", "source", "shape", "opportunity","agreement", "track", "coffee",
    "agent", "ground", "play", "weight", "discipline", "message", "ball", "goal", "dream",
    "customer", "search", "skin", "faith", "exam", "campaign", "material", "theory", "science",
    "distance", "teacher", "principle", "success", "movie", "problem", "son", "instruction",
    "color", "battle", "stranger", "society", "driver", "skill", "failure", "flight", "union",
    "disk", "crisis", "friend", "judge", "nurse", "speech", "income", "agreement", "cell",
    "access", "attack", "trade", "feeling", "protection", "bridge", "frame", "camera", "train",
    "speech", "option", "class", "darkness", "lake", "pressure", "response", "patient", "speech",
    "resource", "client", "poverty", "key", "competition", "queen", "church", "wave", "birth",
    "vehicle", "spirit", "weather", "solution", "garden", "direction", "agency", "glass",
    "future", "bus", "debate", "son", "college", "appearance", "reality", "direction", "feeling",
    "recognition", "assistance", "honor", "drive", "concept", "medium", "memory", "trial",
    "presence", "corner", "region", "breath", "fish", "status", "assistant", "flight", "expense",
    "care", "task", "phase", "painting", "sky", "weekend", "confidence", "talent", "disaster",
    "flower", "feature", "instrument", "sample", "notion", "voice", "profession", "storage",
    "guest", "injury", "expense", "speech", "currency", "balance", "limit", "bridge", "contest",
    "art", "park", "client", "profit", "housing", "comparison", "manufacturer", "pipe", "metal",
    "investment", "boat", "singer", "volume", "colleague", "soil", "crime", "height", "wave",
    "fiction", "employer", "investigation", "suggestion", "flower", "command", "measure",
    "option", "consumer", "principle", "score", "tale", "mission", "operator", "disaster",
    "chocolate", "assistant", "feature", "channel", "direction", "poetry", "independence",
    "ceremony", "drama", "presentation", "abuse", "passage", "wind", "recovery", "faith", "desk",
    "proposal", "profit", "quantity", "investment", "cheek", "attitude", "scheme", "discovery",
    "location", "obligation", "league", "doubt", "corridor", "assessment", "arrival", "mud",
    "intention", "secretary", "dinner", "reflection", "buyer", "gas", "musician", "freedom",
    "breast", "compensation", "climate", "consequence", "protection", "shame", "fiction",
    "judgment", "strike", "training", "feather", "virus", "instruction", "ear", "classroom",
    "baseball", "horse", "apple","recipe", "mixture", "recognition", "justice", "stomach",
    "class", "introduction", "surgery", "writer", "tea", "library", "concept", "menu", "poem",
    "drawer", "camera", "exhibition", "leader", "bread", "patience", "complaint", "woman",
    "storage", "cheese", "wisdom", "ticket", "courage", "independence", "struggle", "procedure",
    "weakness", "birthday", "suggestion", "producer", "departure", "childhood", "companion",
    "departure", "shopping", "cousin", "lab", "ambition", "feeling", "replacement", "dispute",
    "camp", "reflection", "priority", "diet", "arrival", "discovery", "enthusiasm", "king",
    "outcome", "captain", "pollution", "revolution", "reputation", "uncle", "key", "cookie",
    "artist", "slope", "essay", "candle", "development", "basket", "exit", "role", "contribution",
    "apartment", "estate", "category", "sheep", "kingdom", "crew", "engine", "manufacturer",
    "knowledge", "victory", "creation", "furniture", "secret", "disease", "aspect", "midnight",
    "painting", "piano", "cell", "publisher", "wealth", "library", "intention", "desire",
    "contract", "friendship", "breakfast", "pollution", "instrument", "gesture", "actor",
    "confusion", "performance", "mood", "province", "meal", "driver", "consequence", "attitude",
    "soap", "chocolate", "permission", "wealth", "composition", "session", "independence",
    "championship", "curiosity", "complaint", "criticism", "cigarette", "dust", "guest",
    "architect", "intention", "package", "reservation", "dealer", "collector", "recipe",
    "promotion", "laughter", "philosophy", "adventure", "disk", "bottle", "guitar", "contribution",
    "instruction", "tower", "meal", "chicken", "implementation", "intention", "appointment",
    "reflection", "personality", "memory", "transition", "permission", "oven", "entry", "charm",
    "butter", "depth", "examination", "philosophy", "portrait", "migration", "climate", "plate",
    "bread", "tourist", "departure", "kingdom", "employer", "championship", "appointment",
    "pollution", "complication", "advice", "discovery", "elephant", "stomach", "grandmother",
    "skill", "achievement", "discovery", "daughter", "jazz", "cigarette", "suggestion", "vessel",
    "arrival", "leadership", "signature", "girlfriend", "tradition", "singer", "helicopter",
    "airport", "professor", "complaint", "quality", "bedroom", "gallery", "employment",
    "apology", "productivity", "basket", "protection", "profession", "journalist", "movie",
    "sister", "chest", "gate", "hat", "manufacturer", "victory", "shirt", "client"
]

adjectives = [
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
    "right", "big", "high", "different", "small", "large", "next", "early", "young", "important",
    "few", "public", "bad", "same", "able", "last", "best", "better", "high", "old",
    "early", "long", "great", "little", "own", "other", "old", "next", "young", "important",
    "few", "bad", "same", "able", "last", "best", "high", "old", "early", "long",
]

names = [
    "sakura", "kame", "yudai", "qwerty"
]

english_words = list(set(frequent + verbs + noun + adjectives + names))
english_words = [word.lower() for word in english_words]

remove_substrings(english_words)