Ok, here's a more structured outline, incorporating some of your feedback. Let me know what needs to change or is missing.

# Introduction/motivation

- Introduce einsum expressions through examples

- Explain the idea of contracting a tensor op to a binary einsum

- Explain TPPs (small base of highly optimized, widely applicable tensor computations, with certain constraints)

- Explain problem: TPPs assume appropriate data layout for efficiency, ignore data movement. Local lowering doesn't account for layout of preceding contraction

- Brief explanation of the landscape: contraction optimization and TPP scheduling have been approached separately, but we can make progress by solving the two at once.

# The Einsum Trees Solution

- Main contributions:
    - Richer data structure to represent einsum contractions
    - A criterion/algorithm for transforming these trees so as to implement the whole computation using only loops over TPPs.

- Explanation of the dimension taxonomy
- Description of Einsum Tree data structure and optimization algorithm
- A toy example of both the dimension taxonomy and the tree optimization algorithm

- Explain why this structure makes sense; what it allows us to avoid or what it makes faster.

# Performance/Results

- Summarize results: generally better performance than SotA, fraction of the compile time
    - explanation of why the compilation time is better
- Explain evaluation metrics, cost models, etc.

# Evaluation/Outstanding questions

- CPU only?
- Are performance results testing for the right metrics?
- How is the problem solved in other frameworks, how do the solutions differ?
- Any other extensions.
