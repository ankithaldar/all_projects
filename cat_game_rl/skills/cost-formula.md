# Cost Formula Skill

## Formula
Each additional unit n in a batch costs: `init_cost * (1 + 0.5 * (n - 1))`

Closed-form total for batch B: `init_cost * B * (1 + 0.25 * (B - 1))`

## Verification Table

| Batch Size | Multiplier | Example (init_cost=100) |
|-----------|------------|------------------------|
| 1         | 1.00       | 100                    |
| 2         | 1.25       | 250                    |
| 3         | 1.50       | 450                    |
| 5         | 2.00       | 1000                   |
| 10        | 3.25       | 3250                   |
| 20        | 5.75       | 11500                  |

## Inverse Formula
Max affordable batch given coins: `B = floor((-3 + sqrt(9 + 16*coins/init_cost)) / 2)`

## Learnings
- The while-loop approach for the inverse is dangerous due to floating-point precision. Use a single conditional check after the quadratic formula instead.
- Always use `math.ceil()` when converting float cost to int coins to avoid under-charging.
