
nq_prompt = """Write a high-quality answer for the given question using only the provided search results.(some of might be irrelevant).\n\n{contexts}\n\n{question}"""

asqa_prompt = """Given an ambiguous question, figure out its interpretations and answer them one by one. 

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What is the criminal’s name in the breakfast club?
Answer: This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name. In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.

Given an ambiguous question, figure out its interpretations and answer them one by one.\n\n{contexts}\n\nQuestion: {question}\n\nAnswer: """

