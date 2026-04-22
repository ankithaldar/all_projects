# Material Rollback Skill

## Problem
When crafting requires multiple ingredients (e.g., ribbon = 2 string + 2 wood), deducting them sequentially can leave inventory in an inconsistent state if a later ingredient fails.

## Rule
Track all successfully removed ingredients. On any failure, roll back all previously removed:

```python
removed_ings = []
mat_ok = True
for ing in recipe.ingredients:
    success = stash.remove(ing.item_id, ing.quantity * batch)
    if not success:
        for prev_ing in removed_ings:
            stash.add(prev_ing.item_id, prev_ing.quantity * batch)
        mat_ok = False
        break
    removed_ings.append(ing)
```

## Learnings
- Found in audit loop 1: the original `for-else` pattern only rolled back on coin failure, not on ingredient failure
- The `max_affordable_batch_materials()` pre-check should catch most cases, but race conditions between items sharing materials (e.g., two items both needing ribbon) can still trigger partial failures during greedy validation
