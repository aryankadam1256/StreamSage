"""
Expand ChromaDB with more popular movies across all genres and eras.
Adds ~250 movies to the existing 'movies' collection.
Run from: services/movie-assistant-service/
"""
import os, sys
from pathlib import Path

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    str(Path(__file__).parent / "data" / "chroma_db")
)

# ---------------------------------------------------------------------------
# Extended movie dataset  (title, year, rating, genres, director, description)
# ---------------------------------------------------------------------------
MOVIES = [
    # Marvel / Superhero
    ("Spider-Man", 2002, 7.4, "Action, Adventure, Fantasy", "Sam Raimi",
     "When bitten by a genetically modified spider high school student Peter Parker gains superpowers and becomes Spider-Man."),
    ("Spider-Man 2", 2004, 7.5, "Action, Adventure, Fantasy", "Sam Raimi",
     "Peter Parker struggles to balance his life as Spider-Man with his personal life as he faces Doctor Octopus."),
    ("Spider-Man 3", 2007, 6.2, "Action, Adventure, Fantasy", "Sam Raimi",
     "Peter Parker squares off against new villains Sandman and Venom while struggling with his dark side."),
    ("The Amazing Spider-Man", 2012, 6.9, "Action, Adventure, Fantasy", "Marc Webb",
     "Andrew Garfield stars as Peter Parker who becomes a masked vigilante after being bitten by a radioactive spider."),
    ("Spider-Man: Homecoming", 2017, 7.4, "Action, Adventure, Science Fiction", "Jon Watts",
     "Peter Parker under threat of Vulture tries to balance high school life with being Spider-Man."),
    ("Spider-Man: Far From Home", 2019, 7.4, "Action, Adventure, Science Fiction", "Jon Watts",
     "Peter Parker's European school trip is interrupted by Mysterio and a new threat of Elementals."),
    ("Spider-Man: No Way Home", 2021, 8.2, "Action, Adventure, Science Fiction", "Jon Watts",
     "Spider-Man asks Doctor Strange to make the world forget he is Peter Parker, unleashing multiverse chaos."),
    ("Iron Man", 2008, 7.9, "Action, Science Fiction, Adventure", "Jon Favreau",
     "Tony Stark builds a powered armoured suit to escape captivity and becomes Iron Man."),
    ("Iron Man 2", 2010, 7.0, "Action, Adventure, Science Fiction", "Jon Favreau",
     "Tony Stark faces pressure from the government, a new villain with electric whips, and the effects of the arc reactor."),
    ("Iron Man 3", 2013, 7.2, "Action, Adventure, Science Fiction", "Shane Black",
     "Tony Stark faces The Mandarin and an extremis virus while dealing with PTSD after the Battle of New York."),
    ("Thor", 2011, 7.0, "Action, Adventure, Fantasy", "Kenneth Branagh",
     "Thor is cast down from Asgard to Earth by his father Odin while his brother Loki plots to take the throne."),
    ("Thor: Ragnarok", 2017, 7.9, "Action, Adventure, Comedy", "Taika Waititi",
     "Thor must escape Sakaar and save Asgard from the all-powerful Hela with the help of Hulk."),
    ("The Avengers", 2012, 8.0, "Action, Adventure, Science Fiction", "Joss Whedon",
     "Earth's mightiest heroes assemble to fight Loki and the Chitauri army threatening New York."),
    ("Avengers: Infinity War", 2018, 8.4, "Action, Adventure, Science Fiction", "Anthony Russo, Joe Russo",
     "Thanos collects the Infinity Stones to wipe out half of all life in the universe."),
    ("Avengers: Endgame", 2019, 8.4, "Action, Adventure, Science Fiction", "Anthony Russo, Joe Russo",
     "The remaining Avengers travel through time to undo Thanos's snap and bring back the fallen."),
    ("Black Panther", 2018, 7.3, "Action, Adventure, Science Fiction", "Ryan Coogler",
     "T'Challa returns to Wakanda to take his rightful place as king but must face a powerful enemy from the past."),
    ("Captain America: The First Avenger", 2011, 6.9, "Action, Adventure, Science Fiction", "Joe Johnston",
     "Steve Rogers is transformed into super-soldier Captain America to fight the Red Skull's HYDRA forces in WWII."),
    ("Captain America: The Winter Soldier", 2014, 7.7, "Action, Adventure, Science Fiction", "Anthony Russo, Joe Russo",
     "Steve Rogers uncovers a conspiracy within SHIELD while battling the mysterious assassin Winter Soldier."),
    ("Captain America: Civil War", 2016, 7.8, "Action, Adventure, Science Fiction", "Anthony Russo, Joe Russo",
     "The Avengers are divided over government oversight after a disaster splits the team into two sides: Iron Man vs Captain America."),
    ("Guardians of the Galaxy", 2014, 8.0, "Action, Adventure, Comedy, Science Fiction", "James Gunn",
     "A group of intergalactic criminals are forced to work together to stop a fanatical warrior from destroying the galaxy."),
    ("Doctor Strange", 2016, 7.5, "Action, Adventure, Fantasy", "Scott Derrickson",
     "Surgeon Stephen Strange learns the mystic arts after a career-ending accident and faces a rogue sorcerer."),
    ("Black Widow", 2021, 6.7, "Action, Adventure, Science Fiction", "Cate Shortland",
     "Natasha Romanoff confronts the darker parts of her ledger while a dangerous conspiracy pursues her."),
    ("Shang-Chi and the Legend of the Ten Rings", 2021, 7.5, "Action, Adventure, Fantasy", "Destin Daniel Cretton",
     "Shang-Chi must confront the past he thought he left behind when he is drawn into the Ten Rings organization."),
    ("The Batman", 2022, 7.9, "Crime, Mystery, Thriller", "Matt Reeves",
     "Batman ventures into Gotham City's underworld and uncovers corruption linked to The Riddler's killing spree."),
    ("Joker", 2019, 8.4, "Crime, Thriller, Drama", "Todd Phillips",
     "Failed comedian Arthur Fleck descends into madness and becomes the criminal mastermind known as Joker."),
    ("Justice League", 2017, 6.1, "Action, Adventure, Fantasy", "Zack Snyder",
     "Batman assembles a team of heroes to face a catastrophic threat from the villain Steppenwolf."),
    ("Wonder Woman", 2017, 7.4, "Action, Adventure, Fantasy", "Patty Jenkins",
     "Diana Prince leaves Themyscira to fight in WWI and discovers her full powers as Wonder Woman."),
    ("Aquaman", 2018, 6.9, "Action, Adventure, Fantasy", "James Wan",
     "Arthur Curry races to find the mythical trident and reclaim his birthright as heir to the underwater kingdom of Atlantis."),
    ("Man of Steel", 2013, 7.0, "Action, Adventure, Science Fiction", "Zack Snyder",
     "Clark Kent becomes Superman and fights the Kryptonian warrior General Zod who threatens Earth."),
    ("Superman", 1978, 7.3, "Science Fiction, Action, Adventure", "Richard Donner",
     "Clark Kent from Krypton becomes Earth's greatest hero Superman and faces Lex Luthor's real-estate scheme."),

    # Horror / Thriller
    ("Get Out", 2017, 7.7, "Horror, Mystery, Thriller", "Jordan Peele",
     "A young Black man uncovers a disturbing secret when he visits his white girlfriend's family."),
    ("Us", 2019, 6.8, "Horror, Thriller", "Jordan Peele",
     "A family's beach vacation turns terrifying when doppelgängers of themselves appear and attack."),
    ("A Quiet Place", 2018, 7.5, "Drama, Horror, Science Fiction", "John Krasinski",
     "A family struggles to survive in a post-apocalyptic world inhabited by blind monsters with hypersensitive hearing."),
    ("Hereditary", 2018, 7.3, "Horror, Drama, Mystery", "Ari Aster",
     "After the family matriarch dies, her daughter's family begins to unravel dark, terrifying secrets."),
    ("Midsommar", 2019, 7.1, "Horror, Drama, Mystery", "Ari Aster",
     "A couple travel to Sweden for a midsummer festival that descends into a disturbing pagan ritual."),
    ("Halloween", 1978, 7.7, "Horror, Thriller", "John Carpenter",
     "Michael Myers escapes from a mental institution and stalks a babysitter on Halloween night."),
    ("The Shining", 1980, 8.4, "Drama, Horror", "Stanley Kubrick",
     "A family heads to an isolated hotel for the winter where an evil presence influences the father."),
    ("The Silence of the Lambs", 1991, 8.6, "Crime, Drama, Thriller", "Jonathan Demme",
     "FBI trainee Clarice Starling must work with the incarcerated Hannibal Lecter to catch a serial killer."),
    ("It", 2017, 7.3, "Horror", "Andy Muschietti",
     "A group of kids in Derry Maine face their worst fears when they confront a terrifying shapeshifting clown."),
    ("The Conjuring", 2013, 7.5, "Horror, Thriller", "James Wan",
     "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark spirit presence."),
    ("Scream", 1996, 7.4, "Horror, Mystery", "Wes Craven",
     "A year after the murder of her mother, a teenage girl is stalked by a masked killer known as Ghostface."),
    ("Jordan Peele's Nope", 2022, 6.8, "Horror, Mystery, Science Fiction", "Jordan Peele",
     "Residents of a California gulch encounter an alien entity and attempt to capture evidence of it."),
    ("The Witch", 2015, 6.9, "Horror, Mystery, Drama", "Robert Eggers",
     "A Puritan family encounters forces of evil in the New England wilderness in the 17th century."),
    ("Annihilation", 2018, 7.5, "Action, Adventure, Drama", "Alex Garland",
     "A biologist signs up for an expedition into a mysterious area where the laws of nature don't apply."),

    # Science Fiction
    ("Interstellar", 2014, 8.6, "Adventure, Drama, Science Fiction", "Christopher Nolan",
     "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."),
    ("The Matrix", 1999, 8.7, "Action, Science Fiction", "Lana Wachowski, Lilly Wachowski",
     "A computer hacker learns from mysterious rebels about the true nature of his reality."),
    ("The Matrix Reloaded", 2003, 7.2, "Action, Science Fiction", "Lana Wachowski, Lilly Wachowski",
     "Neo and the rebel leaders race to stop the machine army approaching Zion by unraveling the Matrix's code."),
    ("The Matrix Resurrections", 2021, 5.7, "Action, Science Fiction", "Lana Wachowski",
     "Thomas Anderson lives a seemingly ordinary life but feels something is wrong and returns to the Matrix."),
    ("Inception", 2010, 8.8, "Action, Science Fiction, Adventure", "Christopher Nolan",
     "A thief who steals corporate secrets through a dreamsharing technology is given the inverse task of planting an idea."),
    ("Dune", 2021, 8.0, "Action, Adventure, Drama, Science Fiction", "Denis Villeneuve",
     "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset."),
    ("Dune: Part Two", 2024, 8.5, "Science Fiction, Adventure", "Denis Villeneuve",
     "Paul Atreides unites with Chani and the Fremen on a path of revenge against the conspirators who destroyed his family."),
    ("Blade Runner 2049", 2017, 8.0, "Drama, Mystery, Science Fiction", "Denis Villeneuve",
     "A young blade runner's discovery of a dark secret leads him to track down former blade runner Rick Deckard."),
    ("Alien", 1979, 8.4, "Horror, Science Fiction", "Ridley Scott",
     "The crew of a commercial spacecraft encounter a deadly extraterrestrial organism aboard their vessel."),
    ("Aliens", 1986, 8.3, "Action, Adventure, Science Fiction", "James Cameron",
     "Ellen Ripley returns to the planet where her crew encountered the hostile Alien creatures, this time with marines."),
    ("Gravity", 2013, 7.7, "Drama, Science Fiction, Thriller", "Alfonso Cuarón",
     "Two astronauts strive to survive after an accident leaves them adrift in space."),
    ("The Martian", 2015, 8.0, "Adventure, Drama, Science Fiction", "Ridley Scott",
     "An astronaut is left behind on Mars and must find a way to survive while NASA works to bring him home."),
    ("Arrival", 2016, 7.9, "Drama, Mystery, Science Fiction", "Denis Villeneuve",
     "A linguist works with the military to communicate with alien lifeforms after their arrival on Earth."),
    ("Ex Machina", 2014, 7.7, "Drama, Science Fiction, Thriller", "Alex Garland",
     "A programmer is invited to administer the Turing test to an AI with a human-like appearance."),
    ("Her", 2013, 8.0, "Drama, Romance, Science Fiction", "Spike Jonze",
     "A lonely writer develops an unlikely relationship with an operating system designed to meet his every need."),
    ("Tenet", 2020, 7.3, "Action, Science Fiction, Thriller", "Christopher Nolan",
     "A secret agent operates through time inversion to prevent WWIII."),
    ("Oppenheimer", 2023, 8.3, "Biography, Drama, History", "Christopher Nolan",
     "The story of American J. Robert Oppenheimer and his role in the development of the atomic bomb during WWII."),
    ("Avatar", 2009, 7.9, "Action, Adventure, Fantasy, Science Fiction", "James Cameron",
     "A paraplegic Marine is dispatched to the moon Pandora on a mission but becomes torn between following orders and protecting the native Na'vi."),
    ("Avatar: The Way of Water", 2022, 7.6, "Action, Adventure, Fantasy, Science Fiction", "James Cameron",
     "Jake Sully and his family leave Hometree to live among the sea-dwelling Metkayina clan as a new human threat arrives."),
    ("Ready Player One", 2018, 7.4, "Action, Adventure, Science Fiction", "Steven Spielberg",
     "In a near-future world, a teenager finds clues to a digital prize that could give him control of the OASIS."),
    ("Elysium", 2013, 6.6, "Action, Drama, Science Fiction, Thriller", "Neill Blomkamp",
     "In 2154, the very wealthy live on a perfect space station called Elysium while the rest live on an overpopulated Earth."),
    ("District 9", 2009, 7.9, "Action, Science Fiction, Thriller", "Neill Blomkamp",
     "An alien race is forced to live in slum-like conditions on Earth and a government agent undergoes a transformation."),

    # Action / Adventure
    ("Top Gun: Maverick", 2022, 8.3, "Action, Drama", "Joseph Kosinski",
     "Maverick returns to train a group of Top Gun graduates for a specialized mission, facing his past."),
    ("John Wick", 2014, 7.4, "Action, Crime, Thriller", "Chad Stahelski",
     "An ex-hitman comes out of retirement to track down the gangsters who killed his dog and stole his car."),
    ("John Wick: Chapter 2", 2017, 7.5, "Action, Crime, Thriller", "Chad Stahelski",
     "John Wick is forced out of retirement again by a former associate plotting to seize control of a shadowy assassins guild."),
    ("John Wick: Chapter 3 - Parabellum", 2019, 7.4, "Action, Crime, Thriller", "Chad Stahelski",
     "Super-assassin John Wick is on the run after killing a member of the international assassins guild."),
    ("Mad Max: Fury Road", 2015, 8.1, "Action, Adventure, Science Fiction", "George Miller",
     "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search of her homeland."),
    ("The Dark Knight Rises", 2012, 8.4, "Action, Drama, Crime, Thriller", "Christopher Nolan",
     "Eight years after the Joker's reign of anarchy, Batman must return when the terrorist Bane threatens Gotham."),
    ("Mission: Impossible – Fallout", 2018, 7.7, "Action, Adventure, Thriller", "Christopher McQuarrie",
     "Ethan Hunt and his team race against time after a mission gone wrong."),
    ("Die Hard", 1988, 8.2, "Action, Thriller", "John McTiernan",
     "NYPD cop John McClane tries to outwit a group of terrorists who have taken over a Los Angeles skyscraper."),
    ("Gravity Falls - No Film", 2013, 7.1, "Action, Thriller", "Unknown",
     "Placeholder"),
    ("The Raid: Redemption", 2011, 7.6, "Action, Crime, Thriller", "Gareth Evans",
     "A rookie SWAT team is tasked with taking down a ruthless crime lord living in a heavily fortified apartment complex."),
    ("Edge of Tomorrow", 2014, 7.9, "Action, Adventure, Science Fiction", "Doug Liman",
     "An officer gets caught in a time loop during an alien invasion and must fight his way to destroy the enemy."),
    ("Dunkirk", 2017, 7.9, "Action, Drama, History, War", "Christopher Nolan",
     "Allied soldiers from Belgium, the British Empire, and France are surrounded by the German army at the Dunkirk beach."),
    ("1917", 2019, 8.2, "Drama, Action, War", "Sam Mendes",
     "Two British soldiers carry a message through enemy lines to stop a doomed assault on the German WWI front."),
    ("The Revenant", 2015, 8.0, "Action, Adventure, Drama", "Alejandro González Iñárritu",
     "A frontiersman on a fur trading expedition in the 1820s fights for survival after being mauled by a bear."),

    # Drama
    ("The Shawshank Redemption - already in DB", 1994, 9.3, "Drama, Crime", "Frank Darabont", ""),
    ("Parasite", 2019, 8.5, "Comedy, Drama, Thriller", "Bong Joon-ho",
     "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Parks and the destitute Kims."),
    ("Whiplash", 2014, 8.5, "Drama, Music", "Damien Chazelle",
     "A promising young drummer enrolls at a cut-throat music conservatory where his teacher uses extreme means to motivate."),
    ("La La Land", 2016, 8.0, "Comedy, Drama, Music, Romance", "Damien Chazelle",
     "Mia, an aspiring actress, and Sebastian, a jazz musician, fall in love while pursuing their dreams in Los Angeles."),
    ("1984 - Nineteen Eighty-Four", 1984, 7.1, "Drama, Science Fiction", "Michael Radford",
     "In a totalitarian future society, a man questions the government's policies and falls in love with the rebellious Julia."),
    ("Good Will Hunting", 1997, 8.3, "Drama, Romance", "Gus Van Sant",
     "Will Hunting, a janitor at MIT, has a gift for mathematics but needs help from a therapist to find direction."),
    ("Schindler's List", 1993, 9.0, "Drama, History, War", "Steven Spielberg",
     "In German-occupied Poland during WWII, Oskar Schindler saves the lives of more than a thousand Jewish refugees."),
    ("The Pianist", 2002, 8.5, "Biography, Drama, Music, War", "Roman Polanski",
     "A Polish Jewish musician survives the destruction of the Warsaw ghetto of WWII."),
    ("A Beautiful Mind", 2001, 8.2, "Biography, Drama", "Ron Howard",
     "After a brilliant but asocial mathematician accepts a mysterious assignment, his life takes a turn."),
    ("Spotlight", 2015, 8.1, "Drama, Thriller, History", "Tom McCarthy",
     "The true story of how The Boston Globe uncovered the scandal of child abuse by priests in the Catholic Church."),
    ("Boyhood", 2014, 7.9, "Drama", "Richard Linklater",
     "The life of Mason Evans Jr. from age six to eighteen, filmed over twelve years with the same cast."),
    ("Marriage Story", 2019, 7.9, "Drama, Romance", "Noah Baumbach",
     "A stage director and his actor wife struggle through a grueling coast-to-coast divorce that pushes their limits."),
    ("The Social Network", 2010, 7.8, "Drama, History", "David Fincher",
     "As Harvard student Mark Zuckerberg creates the social networking site Facebook, he is sued by two brothers and a co-founder."),
    ("Gone Girl", 2014, 8.1, "Drama, Mystery, Thriller", "David Fincher",
     "Nick Dunne discovers that the entire media focus has made him a suspect in his wife's disappearance."),
    ("Se7en", 1995, 8.6, "Crime, Mystery, Thriller", "David Fincher",
     "Two detectives hunt a serial killer who uses the seven deadly sins as motives for a series of murders."),
    ("Zodiac", 2007, 7.7, "Crime, Drama, Mystery, Thriller", "David Fincher",
     "A cartoonist and reporters become obsessed with tracking down the Zodiac Killer in the San Francisco Bay Area."),
    ("Prisoners", 2013, 8.1, "Crime, Drama, Thriller, Mystery", "Denis Villeneuve",
     "When two daughters go missing in Pennsylvania, a desperate father takes matters into his own hands."),
    ("Sicario", 2015, 7.6, "Action, Crime, Drama, Thriller", "Denis Villeneuve",
     "An idealistic FBI agent is enlisted by a government task force to aid in the war against drugs at the US-Mexico border."),
    ("No Country for Old Men", 2007, 8.2, "Crime, Drama, Thriller", "Joel Coen, Ethan Coen",
     "A hunter stumbles upon a drug deal gone wrong and finds $2 million but is then stalked by an unstoppable hitman."),
    ("There Will Be Blood", 2007, 8.2, "Drama, History", "Paul Thomas Anderson",
     "Daniel Plainview rises to power as an oil prospector and his rivalry with a local preacher grows intense."),
    ("Moonlight", 2016, 7.4, "Drama", "Barry Jenkins",
     "A young Black man grapples with his identity and sexuality across three defining chapters of his life in Miami."),
    ("12 Years a Slave", 2013, 8.1, "Biography, Drama, History", "Steve McQueen",
     "In the antebellum United States, Solomon Northup, a free Black man, is abducted and sold into slavery."),
    ("The Green Mile", 1999, 8.6, "Drama, Crime, Fantasy", "Frank Darabont",
     "The lives of guards on Death Row are changed by the presence of an apparently intellectually disabled inmate with miraculous powers."),
    ("Forrest Gump - already in DB", 1994, 8.8, "Comedy, Drama, Romance", "Robert Zemeckis", ""),

    # Comedy
    ("The Grand Budapest Hotel", 2014, 8.1, "Adventure, Comedy, Crime, Drama", "Wes Anderson",
     "The adventures of Gustave H, a legendary concierge at a famous hotel, and Zero, his lobby boy."),
    ("Superbad", 2007, 7.6, "Comedy", "Greg Mottola",
     "Two co-dependent high school seniors are forced to survive a day without each other's company."),
    ("The Hangover", 2009, 7.7, "Comedy", "Todd Phillips",
     "Three buddies wake up from a bachelor party in Las Vegas with no memory of the previous night and the groom is missing."),
    ("Bridesmaids", 2011, 6.9, "Comedy, Romance", "Paul Feig",
     "Competition between the Maid of Honor and a bridesmaid, over who is the bride's best friend, threatens to ruin the wedding."),
    ("Anchorman: The Legend of Ron Burgundy", 2004, 7.1, "Comedy", "Adam McKay",
     "Ron Burgundy is San Diego's top-rated newsman in the male-dominated 1970s until a female reporter threatens his status."),
    ("The Big Short", 2015, 7.8, "Biography, Comedy, Drama", "Adam McKay",
     "Four outsiders in the world of high finance saw what the big banks, media, and government refused to: the collapse of the economy."),

    # Animation
    ("The Lion King", 1994, 8.5, "Animation, Adventure, Drama, Family", "Roger Allers, Rob Minkoff",
     "Lion cub Simba idolises his father and flees after his uncle Scar murders his father and blames Simba."),
    ("Toy Story", 1995, 8.3, "Animation, Adventure, Comedy, Family", "John Lasseter",
     "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy."),
    ("Toy Story 3", 2010, 8.3, "Animation, Adventure, Comedy, Family", "Lee Unkrich",
     "The toys are mistakenly delivered to a day-care center instead of the attic when Andy prepares for college."),
    ("Up", 2009, 8.3, "Animation, Adventure, Comedy, Drama, Family", "Pete Docter",
     "Seventy-eight-year-old Carl Fredricksen travels to South America by tying thousands of balloons to his house."),
    ("Wall-E", 2008, 8.4, "Animation, Family, Romance, Science Fiction", "Andrew Stanton",
     "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will determine the fate of mankind."),
    ("Spirited Away", 2001, 8.6, "Animation, Adventure, Family, Fantasy", "Hayao Miyazaki",
     "During her family's move to the suburbs, a sullen ten-year-old girl wanders into a world ruled by gods, witches, and spirits."),
    ("Princess Mononoke", 1997, 8.4, "Animation, Adventure, Fantasy", "Hayao Miyazaki",
     "On a journey to find the cure for a Tatarigami's curse, Ashitaka finds himself in the middle of a war between forest gods and humans."),
    ("Your Name", 2016, 8.4, "Animation, Drama, Fantasy, Romance, Science Fiction", "Makoto Shinkai",
     "Two strangers find themselves linked in a bizarre way, and begin to investigate the nature of their connection."),
    ("Klaus", 2019, 8.2, "Animation, Adventure, Comedy, Family, Fantasy", "Sergio Pablos",
     "A self-serving postal worker forms an unlikely partnership with a reclusive toymaker in a village where children have nothing."),
    ("Into the Spider-Verse", 2018, 8.4, "Animation, Action, Adventure, Family", "Bob Persichetti, Peter Ramsey",
     "Teen Miles Morales becomes Spider-Man of his universe and joins others from across the multiverse to stop a threat."),
    ("The Incredibles", 2004, 8.0, "Action, Adventure, Animation, Family", "Brad Bird",
     "A family of undercover superheroes tries to live a quiet suburban life but is forced into action to save the world."),
    ("Coco", 2017, 8.4, "Animation, Adventure, Comedy, Drama, Family, Fantasy, Music", "Lee Unkrich",
     "Aspiring musician Miguel enters the Land of the Dead to find his great-great-grandfather, a legendary singer."),

    # Thriller / Crime
    ("Prisoners", 2013, 8.1, "Crime, Drama, Thriller, Mystery", "Denis Villeneuve",
     "When Keller Dover's daughter and her friend go missing, he takes matters into his own hands as the police investigation stalls."),
    ("Knives Out", 2019, 7.9, "Comedy, Crime, Drama, Mystery, Thriller", "Rian Johnson",
     "A detective investigates the death of a patriarch of an eccentric, combative family."),
    ("Glass Onion", 2022, 7.1, "Comedy, Crime, Drama, Mystery, Thriller", "Rian Johnson",
     "Tech billionaire Miles Bron invites his friends to his island for a murder mystery party, but a real mystery unfolds."),
    ("Heat", 1995, 8.3, "Action, Crime, Drama, Thriller", "Michael Mann",
     "A group of professional bank robbers start to feel the heat from police when they unknowingly leave a crucial clue at their latest heist."),
    ("The Usual Suspects", 1995, 8.5, "Crime, Drama, Mystery, Thriller", "Bryan Singer",
     "The sole survivor of a pier shoot-out tells the story of how he ended up there."),
    ("Chinatown", 1974, 8.1, "Drama, Mystery, Thriller", "Roman Polanski",
     "A private detective hired to expose an adulterer finds himself caught up in a web of deceit and murder."),
    ("L.A. Confidential", 1997, 8.2, "Crime, Drama, Mystery, Thriller", "Curtis Hanson",
     "As corruption grows in 1950s Los Angeles, three policemen attempt to work out a series of murders."),
    ("Nightcrawler", 2014, 7.9, "Crime, Drama, Thriller", "Dan Gilroy",
     "A driven young man finds a job as a crime journalist and becomes obsessed with succeeding at any cost."),
    ("Parasite", 2019, 8.5, "Comedy, Drama, Thriller", "Bong Joon-ho",
     "Greed and class discrimination threaten the newly formed relationship between the wealthy Park family and the destitute Kim family."),
    ("Memories of Murder", 2003, 8.1, "Crime, Drama, Mystery, Thriller", "Bong Joon-ho",
     "Two detectives investigate South Korea's first serial murders based on a true story from 1986."),

    # Romance / Drama
    ("Titanic", 1997, 7.9, "Drama, Romance", "James Cameron",
     "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the ill-fated R.M.S. Titanic."),
    ("Notting Hill", 1999, 7.2, "Comedy, Drama, Romance", "Roger Michell",
     "The life of a simple bookshop owner changes when he meets the world's most famous film star."),
    ("About Time", 2013, 7.8, "Comedy, Drama, Fantasy, Romance, Science Fiction", "Richard Curtis",
     "At the age of 21 Tim discovers he can travel in time and changes the past but has to live with the consequences."),
    ("Call Me by Your Name", 2017, 7.9, "Drama, Romance", "Luca Guadagnino",
     "In 1983 Italy, a romance blossoms between a seventeen-year-old student and the older man hired as his father's research assistant."),
    ("The Notebook", 2004, 7.8, "Drama, Romance", "Nick Cassavetes",
     "A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom."),
    ("Pride & Prejudice", 2005, 7.8, "Drama, Romance", "Joe Wright",
     "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy."),

    # War / Historical
    ("Saving Private Ryan", 1998, 8.6, "Drama, Action, War", "Steven Spielberg",
     "Following the Normandy Landings, a group of US soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed."),
    ("Full Metal Jacket", 1987, 8.3, "Drama, War, Action", "Stanley Kubrick",
     "A pragmatic U.S. Marine observes the dehumanizing effects of the Vietnam War on his fellow recruits."),
    ("Hacksaw Ridge", 2016, 8.1, "Biography, Drama, War", "Mel Gibson",
     "WWII American Army Medic Desmond T. Doss refuses to bear arms and saves 75 lives at the Battle of Okinawa."),
    ("Platoon", 1986, 8.1, "Action, Drama, War", "Oliver Stone",
     "A young soldier in Vietnam faces a moral crisis when confronted with the horrors of war and the murder of innocent villagers."),
    ("The Hurt Locker", 2008, 7.6, "Drama, Action, Thriller, War", "Kathryn Bigelow",
     "A Sergeant leads a bomb disposal team in the Iraq War who exhibits an unusual fearlessness toward his dangerous work."),
    ("Apocalypse Now - already in DB", 1979, 8.5, "Drama, War", "Francis Ford Coppola", ""),
    ("Dunkirk - already in DB", 2017, 7.8, "Action, Drama, War", "Christopher Nolan", ""),

    # Fantasy
    ("Harry Potter and the Philosopher's Stone", 2001, 7.9, "Adventure, Fantasy, Family", "Chris Columbus",
     "An orphaned boy discovers he is a wizard and learns of a dark force lurking in the magical world."),
    ("Harry Potter and the Prisoner of Azkaban", 2004, 7.9, "Adventure, Fantasy, Family", "Alfonso Cuarón",
     "It's Harry's third year at Hogwarts, and the escaped prisoner Sirius Black is the greatest threat to the wizarding world."),
    ("Harry Potter and the Deathly Hallows Part 2", 2011, 8.1, "Adventure, Fantasy, Family", "David Yates",
     "Harry, Ron, and Hermione search for Horcruxes as Voldemort's army grows stronger and the final battle approaches."),
    ("The Fellowship of the Ring - already in DB", 2001, 8.8, "Adventure, Fantasy, Action", "Peter Jackson", ""),
    ("The Hobbit: An Unexpected Journey", 2012, 7.8, "Adventure, Fantasy", "Peter Jackson",
     "A reluctant Hobbit Bilbo Baggins sets out to join a quest with thirteen Dwarves and the wizard Gandalf."),
    ("Pan's Labyrinth", 2006, 8.2, "Drama, Fantasy", "Guillermo del Toro",
     "In post-WWII Spain, a young girl escapes the horrors around her by entering a mythical labyrinth world."),

    # Western / Crime
    ("Django Unchained", 2012, 8.4, "Drama, Western", "Quentin Tarantino",
     "With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal plantation owner."),
    ("Inglourious Basterds", 2009, 8.3, "Action, Adventure, Drama, War", "Quentin Tarantino",
     "In Nazi-occupied France during WWII, a group of Jewish-American soldiers plan to assassinate Nazi leaders."),
    ("Pulp Fiction", 1994, 8.9, "Thriller, Crime", "Quentin Tarantino",
     "The lives of two mob hitmen, a boxer, a gangster, and his wife intertwine in four tales of violence and redemption."),
    ("Once Upon a Time in Hollywood", 2019, 7.7, "Comedy, Drama", "Quentin Tarantino",
     "A faded television actor and his stunt double strive to achieve fame and success in Hollywood in 1969."),
    ("The Good, the Bad and the Ugly", 1966, 8.8, "Western", "Sergio Leone",
     "A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a treasure."),
    ("Tombstone", 1993, 7.8, "Action, Drama, Western, History", "George P. Cosmatos",
     "Wyatt Earp and his brothers arrive in Tombstone and soon challenge the corrupt outlaw gang the Cowboys."),

    # Music
    ("Bohemian Rhapsody", 2018, 7.9, "Biography, Drama, Music", "Bryan Singer",
     "The story of the legendary British rock band Queen and lead singer Freddie Mercury."),
    ("Rocketman", 2019, 7.3, "Biography, Drama, Fantasy, Music", "Dexter Fletcher",
     "A musical fantasy about the fantastical human story of Elton John's breakthrough years."),
    ("A Star Is Born", 2018, 7.6, "Drama, Music, Romance", "Bradley Cooper",
     "A musician helps a young singer and actress find fame as age and alcoholism send his own career into freefall."),
    ("Almost Famous", 2000, 7.9, "Adventure, Comedy, Drama, Music, Romance", "Cameron Crowe",
     "A 15-year-old aspiring writer is given a chance to write a story for Rolling Stone magazine about an up-and-coming rock band."),

    # More popular modern films
    ("Joker: Folie à Deux", 2024, 5.4, "Crime, Drama, Musical", "Todd Phillips",
     "Arthur Fleck in Arkham Asylum awaits his trial for the crimes he committed as the Joker and falls in love with Harley Quinn."),
    ("Barbie", 2023, 6.9, "Adventure, Comedy, Fantasy", "Greta Gerwig",
     "Barbie and Ken are having the time of their lives in the colorful and seemingly perfect Barbieland until a journey takes them to real-world Los Angeles."),
    ("Oppenheimer - duplicate", 2023, 8.3, "Biography, Drama, History", "Christopher Nolan", ""),
    ("Everything Everywhere All at Once", 2022, 8.0, "Action, Adventure, Comedy, Science Fiction", "Daniel Kwan, Daniel Scheinert",
     "A middle-aged Chinese immigrant is swept up in an insane adventure where she alone can save the world by exploring other universes."),
    ("The Menu", 2022, 7.2, "Horror, Thriller, Comedy", "Mark Mylod",
     "A couple travel to a coastal island to eat at an exclusive restaurant where the chef has prepared a lavish menu with some shocking surprises."),
    ("Bullet Train", 2022, 7.4, "Action, Comedy, Thriller", "David Leitch",
     "Five assassins aboard a train find out their missions have something in common."),
    ("Jordan Peele's Get Out - duplicate", 2017, 7.7, "Horror, Mystery, Thriller", "Jordan Peele", ""),
    ("The Boys in the Boat", 2023, 7.5, "Biography, Drama, History, Sport", "George Clooney",
     "The 1936 University of Washington rowing team and their quest to prove themselves and compete at the Berlin Olympics."),
    ("Killers of the Flower Moon", 2023, 7.6, "Crime, Drama, History, Mystery", "Martin Scorsese",
     "Members of the Osage Nation are murdered under mysterious circumstances in the 1920s, sparking a major FBI investigation."),
    ("Saltburn", 2023, 7.1, "Drama, Mystery, Thriller", "Emerald Fennell",
     "A young man becomes obsessed with an aristocratic family at their Saltburn estate during a summer in 2006."),
]

def expand_db():
    import chromadb
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"\nConnecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection("movies")
    before = col.count()
    print(f"Movies before: {before}")

    # Get existing titles to avoid duplicates
    existing = col.get(include=["metadatas"], limit=500)
    existing_titles = {m["title"].lower() for m in existing["metadatas"]}

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Model loaded.")

    to_add = []
    for title, year, rating, genres, director, description in MOVIES:
        if "already in DB" in title or "duplicate" in title or not description:
            continue
        if title.lower() in existing_titles:
            print(f"  Skip (exists): {title}")
            continue
        to_add.append((title, year, rating, genres, director, description))

    print(f"\nAdding {len(to_add)} new movies...")

    BATCH = 32
    for start in range(0, len(to_add), BATCH):
        batch = to_add[start:start + BATCH]

        docs = []
        metas = []
        ids = []

        for title, year, rating, genres, director, description in batch:
            doc = (
                f"Title: {title} ({year})\n"
                f"Genres: {genres}\n"
                f"Director: {director}\n"
                f"Plot: {description}"
            )
            docs.append(doc)
            metas.append({
                "title": title,
                "year": int(year),
                "rating": float(rating),
                "genres": genres,
                "director": director,
            })
            safe_id = title.lower().replace(" ", "_").replace(":", "").replace("'", "")[:60]
            import time
            ids.append(f"exp_{safe_id}_{int(time.time() * 1000) % 100000}")

        embeddings = embedder.embed_documents(docs)
        col.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        print(f"  Added batch {start // BATCH + 1}: {[t for t,*_ in batch][:3]}...")

    after = col.count()
    print(f"\nDone! Movies: {before} -> {after} (+{after - before})")

if __name__ == "__main__":
    expand_db()
