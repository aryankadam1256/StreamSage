"""
100-Query Evaluation Suite for Movie Discovery Assistant
========================================================
Runs 100 diverse real-world user queries, records results,
and grades them against expected ideal answers.

Grading:
  Excellent  — 4+ of top 5 are clearly relevant, well-known, correct
  Good       — 2-3 of top 5 are relevant
  Weak       — 1 of top 5 is relevant, rest are off-topic or low-quality
  Terrible   — 0 relevant results, or results contradict the query intent
"""

import urllib.request
import json
import time
import sys

API = "http://localhost:8004/discover"

# ============================================================================
# 100 TEST QUERIES — grouped by category
# Each tuple: (query, top_k, expected_description, must_include_keywords, must_not_include_keywords)
#   must_include  = at least ONE of these words should appear in results (titles/genres)
#   must_not      = NONE of these should dominate results
# ============================================================================

TESTS = [
    # ── CATEGORY 1: Director Queries (10) ──
    ("Martin Scorsese films", 5,
     "Goodfellas, Casino, The Departed, Taxi Driver, Raging Bull",
     ["goodfellas","departed","casino","taxi driver","raging bull","wolf","shutter island"],
     ["documentary"]),
    ("Steven Spielberg movies", 5,
     "Schindler's List, Jaws, E.T., Raiders, Saving Private Ryan",
     ["schindler","jaws","e.t","raiders","saving private","jurassic"],
     ["documentary"]),
    ("Christopher Nolan best films", 5,
     "Inception, Interstellar, Dark Knight, Memento, Oppenheimer",
     ["inception","interstellar","dark knight","memento","oppenheimer","prestige"],
     ["documentary"]),
    ("Quentin Tarantino movies", 5,
     "Pulp Fiction, Kill Bill, Django, Inglourious Basterds",
     ["pulp fiction","kill bill","django","inglourious","reservoir"],
     ["documentary"]),
    ("James Cameron movies", 5,
     "Terminator, Aliens, Titanic, Avatar",
     ["terminator","aliens","titanic","avatar"],
     ["documentary"]),
    ("Wes Anderson films", 5,
     "Grand Budapest, Moonrise Kingdom, Royal Tenenbaums",
     ["grand budapest","moonrise","tenenbaums","fantastic mr","darjeeling"],
     ["documentary"]),
    ("David Fincher films", 5,
     "Fight Club, Se7en, Zodiac, Social Network, Gone Girl",
     ["fight club","se7en","seven","zodiac","social network","gone girl"],
     ["documentary"]),
    ("Ridley Scott movies", 5,
     "Gladiator, Blade Runner, Alien, The Martian",
     ["gladiator","blade runner","alien","martian"],
     ["documentary"]),
    ("Stanley Kubrick films", 5,
     "2001, The Shining, A Clockwork Orange, Full Metal Jacket",
     ["2001","shining","clockwork","full metal","eyes wide"],
     ["documentary"]),
    ("Denis Villeneuve movies", 5,
     "Dune, Arrival, Blade Runner 2049, Sicario, Prisoners",
     ["dune","arrival","blade runner 2049","sicario","prisoners"],
     ["documentary"]),

    # ── CATEGORY 2: Genre Queries (15) ──
    ("best horror movies", 5,
     "The Exorcist, Hereditary, The Shining, Get Out",
     ["exorcist","hereditary","shining","get out","conjuring","ring","halloween"],
     []),
    ("romantic comedy", 5,
     "When Harry Met Sally, Notting Hill, Pretty Woman, 10 Things",
     ["when harry","notting hill","pretty woman","bridget","love actually"],
     ["horror","thriller"]),
    ("psychological thriller", 5,
     "Shutter Island, Black Swan, Silence of the Lambs, Gone Girl",
     ["shutter island","black swan","silence","gone girl","se7en"],
     []),
    ("war movies", 5,
     "Saving Private Ryan, Apocalypse Now, Dunkirk, Platoon",
     ["saving private","apocalypse","dunkirk","platoon","1917","hacksaw"],
     []),
    ("animated movies for kids", 5,
     "Finding Nemo, Toy Story, Frozen, Lion King, Inside Out",
     ["nemo","toy story","frozen","lion king","inside out","moana","coco"],
     ["horror","thriller"]),
    ("heist movies", 5,
     "Ocean's Eleven, The Italian Job, Heat, Baby Driver",
     ["ocean","italian job","heat","baby driver","inside man","sting"],
     []),
    ("disaster movies", 5,
     "The Day After Tomorrow, 2012, San Andreas, Twister",
     ["day after","2012","san andreas","twister","independence day","armageddon"],
     []),
    ("sports movies", 5,
     "Rocky, Moneyball, Remember the Titans, Creed",
     ["rocky","moneyball","remember","creed","miracle","rudy","raging bull"],
     []),
    ("western movies", 5,
     "The Good the Bad the Ugly, Unforgiven, Django, True Grit",
     ["good bad ugly","unforgiven","django","true grit","tombstone","magnificent"],
     []),
    ("musical films", 5,
     "La La Land, Singin' in the Rain, West Side Story, Chicago",
     ["la la land","singin","west side","chicago","greatest showman","grease"],
     []),
    ("zombie movies", 5,
     "28 Days Later, Shaun of the Dead, World War Z, Train to Busan",
     ["28 days","shaun","world war z","train to busan","night of the living","dawn"],
     []),
    ("superhero movies", 5,
     "The Dark Knight, Spider-Man, Avengers, Logan",
     ["dark knight","spider","avengers","logan","iron man","batman","superman"],
     []),
    ("space movies", 5,
     "Interstellar, Gravity, The Martian, 2001, Alien",
     ["interstellar","gravity","martian","2001","alien","apollo","moon"],
     []),
    ("courtroom drama", 5,
     "12 Angry Men, A Few Good Men, To Kill a Mockingbird",
     ["12 angry","few good men","mockingbird","verdict","primal fear"],
     []),
    ("mafia movies", 5,
     "The Godfather, Goodfellas, Casino, Scarface, The Departed",
     ["godfather","goodfellas","casino","scarface","departed"],
     ["documentary"]),

    # ── CATEGORY 3: Mood / Emotion Queries (15) ──
    ("feel good movies", 5,
     "Forrest Gump, The Intouchables, Amelie, Up",
     ["forrest gump","intouchables","amelie","up","pursuit","secret life"],
     ["horror","thriller"]),
    ("movies to watch when sad", 5,
     "Inside Out, Paddington, The Secret Life of Walter Mitty — comforting",
     ["inside out","paddington","secret life","soul","chef"],
     []),
    ("movies that make you cry", 5,
     "Schindler's List, The Green Mile, Hachi, Titanic",
     ["schindler","green mile","hachi","titanic","notebook","fault"],
     []),
    ("scary movies for halloween", 5,
     "Halloween, The Conjuring, It, A Nightmare on Elm Street",
     ["halloween","conjuring","it","nightmare","scream","exorcist"],
     ["comedy","romance"]),
    ("funny movies to watch with friends", 5,
     "Superbad, The Hangover, Step Brothers, Bridesmaids",
     ["superbad","hangover","step brothers","bridesmaids","21 jump","anchorman"],
     []),
    ("dark and disturbing films", 5,
     "Requiem for a Dream, A Clockwork Orange, Se7en, Oldboy",
     ["requiem","clockwork","se7en","seven","oldboy","no country"],
     []),
    ("uplifting inspirational movies", 5,
     "Rocky, The Pursuit of Happyness, Dead Poets Society, Good Will",
     ["rocky","pursuit","dead poets","good will","shawshank"],
     []),
    ("rainy day movies", 5,
     "Cozy comfortable films — Amelie, Paddington, Chef",
     ["amelie","paddington","chef","julie","grand budapest"],
     ["horror","slasher"]),
    ("movies for date night", 5,
     "La La Land, Titanic, The Notebook, Crazy Rich Asians",
     ["la la land","titanic","notebook","crazy rich","before sunrise"],
     []),
    ("edge of your seat thrillers", 5,
     "No Country for Old Men, Sicario, Nightcrawler, Se7en",
     ["no country","sicario","nightcrawler","se7en","zodiac","prisoners"],
     []),
    ("movies that make you think", 5,
     "Inception, Arrival, Ex Machina, Eternal Sunshine, Matrix",
     ["inception","arrival","ex machina","eternal sunshine","matrix","interstellar"],
     []),
    ("light hearted comedy", 5,
     "Ferris Bueller, The Grand Budapest Hotel, School of Rock",
     ["ferris","grand budapest","school of rock","juno","mean girls"],
     ["horror","war"]),
    ("adrenaline rush action movies", 5,
     "Mad Max Fury Road, John Wick, Die Hard, Mission Impossible",
     ["mad max","john wick","die hard","mission impossible","fast"],
     []),
    ("cheer me up", 5,
     "Happy comfortable films — Legally Blonde, Paddington, School of Rock",
     ["legally blonde","paddington","school of rock","princess","elf"],
     ["horror","war","sad"]),
    ("bittersweet movies", 5,
     "Lost in Translation, Eternal Sunshine, Her, Manchester by the Sea",
     ["lost in translation","eternal sunshine","her","manchester","moonlight"],
     []),

    # ── CATEGORY 4: Decade / Era Queries (8) ──
    ("best movies of 2023", 5,
     "Oppenheimer, Barbie, Killers of the Flower Moon, Guardians 3",
     ["oppenheimer","barbie","killers","guardians","john wick"],
     []),
    ("classic 80s movies", 5,
     "Back to the Future, E.T., The Breakfast Club, Ghostbusters",
     ["back to the future","e.t","breakfast club","ghostbusters","blade runner","aliens"],
     []),
    ("90s nostalgia movies", 5,
     "Pulp Fiction, Titanic, The Matrix, Fight Club, Jurassic Park",
     ["pulp fiction","titanic","matrix","fight club","jurassic","forrest gump"],
     []),
    ("classic 1950s films", 5,
     "12 Angry Men, Rear Window, Singin' in the Rain, Ben-Hur",
     ["12 angry","rear window","singin","ben-hur","vertigo","sunset"],
     []),
    ("best movies of the 70s", 5,
     "The Godfather, Jaws, Star Wars, Taxi Driver, Apocalypse Now",
     ["godfather","jaws","star wars","taxi driver","apocalypse","alien"],
     []),
    ("2010s masterpieces", 5,
     "Inception, Parasite, Mad Max Fury Road, Whiplash",
     ["inception","parasite","mad max","whiplash","interstellar","moonlight"],
     []),
    ("old black and white movies", 5,
     "Casablanca, It's a Wonderful Life, Citizen Kane, 12 Angry Men",
     ["casablanca","wonderful life","citizen kane","12 angry","psycho"],
     []),
    ("recent movies from 2024", 5,
     "Dune Part Two, Furiosa, Inside Out 2, Deadpool & Wolverine",
     ["dune","furiosa","inside out","deadpool","gladiator"],
     []),

    # ── CATEGORY 5: "Movies Like X" (Similarity) (12) ──
    ("movies like Inception", 5,
     "Interstellar, The Matrix, Shutter Island, Memento",
     ["interstellar","matrix","shutter island","memento","prestige","dark city"],
     []),
    ("something similar to The Godfather", 5,
     "Goodfellas, Scarface, Casino, The Departed, Once Upon a Time in America",
     ["goodfellas","scarface","casino","departed","once upon"],
     []),
    ("movies like Toy Story", 5,
     "Finding Nemo, Inside Out, Monsters Inc, Shrek, Up",
     ["nemo","inside out","monsters","shrek","up","incredibles"],
     ["horror"]),
    ("films similar to Pulp Fiction", 5,
     "Reservoir Dogs, Snatch, Lock Stock, Kill Bill, Fight Club",
     ["reservoir","snatch","lock stock","kill bill","fight club"],
     []),
    ("movies like The Notebook", 5,
     "A Walk to Remember, Me Before You, The Fault in Our Stars, P.S. I Love You",
     ["walk to remember","me before you","fault in our stars","p.s. i love","dear john"],
     ["horror","action"]),
    ("something like Interstellar", 5,
     "Arrival, Gravity, The Martian, 2001, Contact",
     ["arrival","gravity","martian","2001","contact","moon"],
     []),
    ("movies similar to John Wick", 5,
     "Atomic Blonde, Nobody, Equalizer, Taken, Kill Bill",
     ["atomic blonde","nobody","equalizer","taken","kill bill"],
     []),
    ("films like Spirited Away", 5,
     "Howl's Moving Castle, My Neighbor Totoro, Princess Mononoke",
     ["howl","totoro","mononoke","ponyo","kiki"],
     []),
    ("movies like Get Out", 5,
     "Us, Midsommar, The Invitation, Don't Breathe, Nope",
     ["us","midsommar","invitation","don't breathe","nope"],
     []),
    ("something like The Dark Knight", 5,
     "Batman Begins, Logan, Watchmen, V for Vendetta",
     ["batman begins","logan","watchmen","v for vendetta","spider-man"],
     []),
    ("movies like Forrest Gump", 5,
     "The Curious Case of Benjamin Button, Big Fish, Cast Away",
     ["benjamin button","big fish","cast away","green mile","pursuit"],
     []),
    ("films similar to Parasite", 5,
     "Snowpiercer, Shoplifters, Us, Burning, Knives Out",
     ["snowpiercer","shoplifters","us","burning","knives out"],
     []),

    # ── CATEGORY 6: Actor Queries (8) ──
    ("Leonardo DiCaprio best movies", 5,
     "Inception, Wolf of Wall Street, The Revenant, Shutter Island",
     ["inception","wolf","revenant","shutter island","departed","gatsby"],
     ["documentary"]),
    ("Tom Hanks movies", 5,
     "Forrest Gump, Cast Away, Saving Private Ryan, The Green Mile",
     ["forrest gump","cast away","saving private","green mile","philadelphia"],
     ["documentary"]),
    ("Scarlett Johansson films", 5,
     "Lost in Translation, Marriage Story, Her, Black Widow, Lucy",
     ["lost in translation","marriage story","her","black widow","lucy"],
     ["documentary"]),
    ("Brad Pitt movies", 5,
     "Fight Club, Se7en, Inglourious Basterds, Once Upon a Time",
     ["fight club","se7en","seven","inglourious","once upon","ocean","moneyball"],
     ["documentary"]),
    ("Morgan Freeman movies", 5,
     "Shawshank Redemption, Se7en, Million Dollar Baby, Driving Miss Daisy",
     ["shawshank","se7en","seven","million dollar","driving miss"],
     ["documentary"]),
    ("Keanu Reeves films", 5,
     "The Matrix, John Wick, Speed, Point Break",
     ["matrix","john wick","speed","point break","bill & ted"],
     ["documentary"]),
    ("Meryl Streep movies", 5,
     "The Devil Wears Prada, Sophie's Choice, Kramer vs Kramer",
     ["devil wears prada","sophie","kramer","iron lady","bridges"],
     ["documentary"]),
    ("Denzel Washington films", 5,
     "Training Day, Glory, Malcolm X, Fences, Man on Fire",
     ["training day","glory","malcolm","fences","man on fire","equalizer"],
     ["documentary"]),

    # ── CATEGORY 7: Abstract / Conceptual (12) ──
    ("hidden gems", 5,
     "Underrated films — Moon, The Fall, Coherence, Predestination",
     ["moon","coherence","predestination","primer","waking life"],
     []),
    ("movies with unexpected twist endings", 5,
     "The Sixth Sense, The Usual Suspects, Fight Club, Oldboy",
     ["sixth sense","usual suspects","fight club","oldboy","prestige","shutter"],
     []),
    ("visually stunning cinematography", 5,
     "Blade Runner 2049, The Revenant, Hero, Life of Pi",
     ["blade runner 2049","revenant","hero","life of pi","tree of life","grand budapest"],
     []),
    ("underrated movies you haven't seen", 5,
     "Spotlight, Nightcrawler, Arrival, Whiplash",
     ["spotlight","nightcrawler","arrival","whiplash","prisoners","moon"],
     []),
    ("mind bending movies", 5,
     "Inception, The Matrix, Memento, Predestination, Primer",
     ["inception","matrix","memento","predestination","primer","donnie darko"],
     []),
    ("movies based on true story", 5,
     "Schindler's List, The Social Network, 12 Years a Slave",
     ["schindler","social network","12 years","wolf","imitation","spotlight"],
     []),
    ("movies with great plot twists", 5,
     "The Sixth Sense, Fight Club, The Prestige, Shutter Island",
     ["sixth sense","fight club","prestige","shutter island","gone girl","usual"],
     []),
    ("coming of age movies", 5,
     "Stand by Me, Lady Bird, The Perks, Boyhood, Juno",
     ["stand by me","lady bird","perks","boyhood","juno","breakfast club"],
     []),
    ("road trip movies", 5,
     "Little Miss Sunshine, Into the Wild, Thelma & Louise",
     ["little miss","into the wild","thelma","rain man","sideways"],
     []),
    ("movies about time travel", 5,
     "Back to the Future, Interstellar, Looper, Primer, Edge of Tomorrow",
     ["back to the future","interstellar","looper","primer","edge of tomorrow","12 monkeys"],
     []),
    ("black comedy", 5,
     "Fargo, In Bruges, Dr. Strangelove, Parasite, Pulp Fiction",
     ["fargo","in bruges","strangelove","parasite","pulp fiction","burn after"],
     []),
    ("movies about artificial intelligence", 5,
     "Ex Machina, Her, 2001, Blade Runner, The Matrix, A.I.",
     ["ex machina","her","2001","blade runner","matrix","a.i.","terminator"],
     []),

    # ── CATEGORY 8: Specific Requests (10) ──
    ("movies to watch with grandma", 5,
     "Wholesome gentle films — Mrs Doubtfire, Cocoon, Driving Miss Daisy",
     ["doubtfire","cocoon","driving miss","sound of music","forrest gump"],
     ["horror","slasher","violence"]),
    ("good movies for a 10 year old", 5,
     "Harry Potter, Frozen, Finding Nemo, The Incredibles",
     ["harry potter","frozen","nemo","incredibles","lion king","moana"],
     ["horror","r-rated"]),
    ("foreign language must watch", 5,
     "Parasite, Amelie, Pan's Labyrinth, City of God, Spirited Away",
     ["parasite","amelie","pan's labyrinth","city of god","spirited away","oldboy"],
     []),
    ("oscar best picture winners", 5,
     "Parasite, Schindler's List, The Godfather, No Country",
     ["parasite","schindler","godfather","no country","moonlight","green book"],
     []),
    ("movies with great soundtracks", 5,
     "Interstellar, Inception, Pulp Fiction, Guardians of the Galaxy",
     ["interstellar","inception","pulp fiction","guardians","baby driver","drive"],
     []),
    ("three hour epic movies", 5,
     "Lord of the Rings, The Godfather, Schindler's List, Lawrence of Arabia",
     ["lord of the rings","godfather","schindler","lawrence","braveheart"],
     []),
    ("movies everyone should watch before they die", 5,
     "Shawshank, The Godfather, Schindler's List, Pulp Fiction",
     ["shawshank","godfather","schindler","pulp fiction","citizen kane","12 angry"],
     []),
    ("best movie trilogies", 5,
     "Lord of the Rings, The Dark Knight, Star Wars, Godfather, Back to Future",
     ["lord of the rings","dark knight","star wars","godfather","back to the future"],
     []),
    ("slow burn thriller", 5,
     "Zodiac, No Country for Old Men, Tinker Tailor, Prisoners",
     ["zodiac","no country","tinker tailor","prisoners","nightcrawler","gone girl"],
     []),
    ("movies set in New York", 5,
     "Taxi Driver, Breakfast at Tiffany's, Spider-Man, Ghostbusters",
     ["taxi driver","breakfast","spider-man","ghostbusters","manhattan","king of comedy"],
     []),

    # ── CATEGORY 9: Franchise / Series (5) ──
    ("Harry Potter movies", 5,
     "All Harry Potter films",
     ["harry potter","philosopher","chamber","azkaban","goblet","phoenix","prince","deathly"],
     []),
    ("Marvel Avengers movies", 5,
     "Avengers, Age of Ultron, Infinity War, Endgame",
     ["avengers","ultron","infinity war","endgame"],
     []),
    ("Star Wars movies", 5,
     "Original trilogy, prequels, or sequels",
     ["star wars","empire strikes","return of the jedi","phantom","force awakens"],
     []),
    ("Lord of the Rings", 5,
     "Fellowship, Two Towers, Return of the King, The Hobbit",
     ["fellowship","two towers","return of the king","hobbit"],
     []),
    ("Batman movies", 5,
     "The Dark Knight, Batman Begins, The Batman, Batman Returns",
     ["dark knight","batman begins","batman","returns","forever"],
     ["documentary"]),

    # ── CATEGORY 10: Conversational / Natural Language (5) ──
    ("I want something like Breaking Bad but as a movie", 5,
     "Crime dramas with antiheroes — No Country, Sicario, Traffic",
     ["no country","sicario","traffic","scarface","departed","nightcrawler"],
     []),
    ("what's a good movie to watch tonight", 5,
     "Broadly accessible popular films",
     [],  # any good movie is fine
     []),
    ("my girlfriend likes romance and I like action", 5,
     "Action-romance blend — Mr and Mrs Smith, True Lies, The Princess Bride",
     ["mr and mrs smith","true lies","princess bride","knight","day"],
     []),
    ("recommend me something I've never heard of", 5,
     "Obscure/cult picks — Coherence, The Fall, Moon, Predestination",
     [],  # hard to validate, just check quality
     []),
    ("best movies of all time", 5,
     "Shawshank, Godfather, Schindler's, Dark Knight, Pulp Fiction",
     ["shawshank","godfather","schindler","dark knight","pulp fiction","12 angry"],
     []),
]


def run_test(query, top_k, expected, must_include, must_not):
    """Run a single query and grade the results."""
    payload = json.dumps({"query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(API, data=payload,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"query": query, "error": str(e), "grade": "ERROR", "movies": [], "expected": expected}

    movies = data.get("recommended_movies", [])
    titles = [m["title"] for m in movies]
    titles_lower = " | ".join(titles).lower()
    all_genres = " ".join(m.get("genres", "") for m in movies).lower()

    # Grade: count how many expected keywords appear in results
    hits = sum(1 for kw in must_include if kw in titles_lower)
    hit_pct = hits / max(len(must_include), 1)

    # Check must-not violations
    violations = [kw for kw in must_not if kw in all_genres and kw not in query.lower()]

    # Grade
    if len(must_include) == 0:
        # Can't auto-grade — check ratings
        avg_rating = sum(float(m.get("rating", 0) or 0) for m in movies) / max(len(movies), 1)
        if avg_rating >= 7.0:
            grade = "Excellent"
        elif avg_rating >= 6.0:
            grade = "Good"
        else:
            grade = "Weak"
    elif violations and hit_pct < 0.3:
        grade = "Terrible"
    elif hit_pct >= 0.5:
        grade = "Excellent"
    elif hit_pct >= 0.3:
        grade = "Good"
    elif hit_pct >= 0.15:
        grade = "Weak"
    else:
        grade = "Terrible"

    return {
        "query": query,
        "movies": [
            {"title": m["title"], "year": m.get("year"), "rating": m.get("rating"),
             "genres": m.get("genres", "")}
            for m in movies
        ],
        "expected": expected,
        "grade": grade,
        "hits": hits,
        "total_expected": len(must_include),
        "violations": violations,
    }


def main():
    print(f"Running {len(TESTS)} queries against http://localhost:8004 ...")
    results = []
    grades = {"Excellent": 0, "Good": 0, "Weak": 0, "Terrible": 0, "ERROR": 0}
    categories = {}

    for i, (query, top_k, expected, must_inc, must_not) in enumerate(TESTS):
        sys.stdout.write(f"\r  [{i+1}/{len(TESTS)}] {query[:50]}...")
        sys.stdout.flush()
        r = run_test(query, top_k, expected, must_inc, must_not)
        results.append(r)
        grades[r["grade"]] = grades.get(r["grade"], 0) + 1

        # Determine category from position
        if i < 10: cat = "Director"
        elif i < 25: cat = "Genre"
        elif i < 40: cat = "Mood/Emotion"
        elif i < 48: cat = "Decade/Era"
        elif i < 60: cat = "Similar-To"
        elif i < 68: cat = "Actor"
        elif i < 80: cat = "Abstract/Concept"
        elif i < 90: cat = "Specific Request"
        elif i < 95: cat = "Franchise"
        else: cat = "Conversational"

        r["category"] = cat
        if cat not in categories:
            categories[cat] = {"Excellent": 0, "Good": 0, "Weak": 0, "Terrible": 0, "ERROR": 0}
        categories[cat][r["grade"]] += 1

    print("\n")

    # ── Summary ──
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    total = len(TESTS)
    for g in ["Excellent", "Good", "Weak", "Terrible", "ERROR"]:
        bar = "#" * grades[g]
        print(f"  {g:10s}: {grades[g]:3d}/{total}  {bar}")
    print()

    # ── Per-Category ──
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 70)
    for cat, cg in sorted(categories.items()):
        cat_total = sum(cg.values())
        exc = cg["Excellent"]
        good = cg["Good"]
        weak = cg["Weak"]
        terr = cg["Terrible"]
        err = cg["ERROR"]
        print(f"  {cat:20s}: {exc}E {good}G {weak}W {terr}T {err}ERR  (/{cat_total})")
    print()

    # ── Failures ──
    print("WEAK / TERRIBLE QUERIES (need improvement)")
    print("-" * 70)
    for r in results:
        if r["grade"] in ("Weak", "Terrible", "ERROR"):
            titles = ", ".join(m["title"] for m in r.get("movies", []))
            print(f"\n  [{r['grade']}] {r['query']}")
            print(f"    Got:      {titles}")
            print(f"    Expected: {r['expected']}")
            if r.get("violations"):
                print(f"    Violations: {r['violations']}")
            print(f"    Hits: {r.get('hits',0)}/{r.get('total_expected',0)}")

    # ── Save full results to JSON ──
    output = {
        "summary": grades,
        "categories": categories,
        "results": results,
    }
    with open("test_100_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to test_100_results.json")


if __name__ == "__main__":
    main()
