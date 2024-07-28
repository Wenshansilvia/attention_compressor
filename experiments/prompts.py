
nq_prompt = """Write a high-quality answer for the given question using only the provided search results.(some of might be irrelevant).\n\n{contexts}\n\n{question}"""

asqa_prompt = """Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: Who played bonnie in gone with the wind? 
Answer: This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to. In order to figure out its interpretations, we need to consider different versions or adaptations of Gone with the Wind. Gone with the Wind has two versions or adaptations: the 1939 film Gone with the Wind or the 2008 musical Gone with the Wind. Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind’s character Bonnie was played by Eleanore Cammack "Cammie" King. (2) The 2008 musical Gone with the Wind’s character Bonnie was played by Leilah de Meza. 

Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: What is the second largest city in the usa? 
Answer: This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA. In order to figure out its interpretations, we need to consider different criteria to determine a city’s size. City size can be measured by two criteria: population or area. Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska. 

Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: When was bohemian rhapsody released as a single? 
Answer: This question is ambiguous in terms of which country’s release of the single is being referred to. In order to figure out its interpretations, we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates. Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975. 

Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: Where do the philadelphia eagles play their home games? 
Answer: This question is ambiguous in terms of which specific location or venue is being referred to. In order to figure out its interpretations, we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium. 

Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: When did xbox one come out in australia? 
Answer: This question is ambiguous in terms of which specific version of the Xbox One is being referred to. In order to figure out its interpretations, we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model. Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017. 

Given an ambiguous question, figure out its interpretations and answer them one by one. 
Question: When does the movie summer of 84 come out? 
Answer: This question is ambiguous in terms of which release of the movie is being referred to. In order to figure out its interpretations, we need to consider different releases of the movie Summer of ’84. The movie Summer of ’84 is first released at the Sundance Festival before it’s released throughout the US. Therefore, this question has 2 interpretations: (1) When did the movie Summer of ’84 first release at the Sundance Festival? (2) When did the movie Summer of ’84 first release throughout the US? The answers to all interpretations are: (1) Summer of ’84 was released at the Sundance Festival on January 22, 2018. (2) Summer of ’84 was released throughout the US on August 10, 2018.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What was roy orbison’s first number one hit?
Answer: This question is ambiguous in terms of which specific chart or region is being referred to. In order to figure out its interpretations, we need to consider the different charts and regions where Roy Orbison’s music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland. Therefore, this question has 2 interpretations: (1) What was Roy Orbison’s first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison’s first number one hit in the UK and Ireland? The answers to all interpretations are: (1) Running Scared was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) Only the Lonely (Know the Way I Feel) was the first number one hit for Roy Orbison in the UK and Ireland.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What is the criminal’s name in the breakfast club?
Answer: This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name. In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.

Given an ambiguous question, figure out its interpretations and answer them one by one.\n\n{contexts}\n\nQuestion: {question}\n\nAnswer: """

