# RPLAN inputs for APM (Table 1 block (a)): landing and hallway

Answers to two questions from Fedor (2026-11-05) about the 1,000 programs in
`noface:/data/mason/michael/benchmark_inputs/rplan_g2p_test1000/` (written by `scripts/rplan/export_mason_programs.py`).

## 1. Why is there a `landing` in front of the entrance?

Mason2 requires an outdoor node at the front door. `docs/concepts/01-home-state.md` (mason2): `entrance` is a door
opening that must be "exactly one door between an indoor and an outdoor room", and "every home has at least one real
outdoor room ... or an entrance landing". RPLAN has nothing outside the front door, so the export adds
`landing:0` (2 m², `synthetic: true`, entrance room -- landing door), the same node Mason2's own homes have
(e.g. `next-unified-storey-a9f09e7f5803` and `next-multistory-nav-4648dcf51a12`: `landing:0`, 2.025 m², entrance
`opening:edge:hallway:0--landing:0`).

It does not affect the scores: `scripts/ours2/convert.py` drops the landing and turns the door to it into the front
door; Types, Graph and Area are scored against the RPLAN input, which has no landing.

It could be dropped from the inputs if APM derives the landing itself (Mason2's explicit-wall helper already derives
entrance landings, `docs/concepts/04-stages.md`). The numbers would not change.

## 2. Is the hallway really missing, or is it a naming issue?

It is really missing. RPLAN's 13 room classes are LivingRoom, MasterRoom, Kitchen, Bathroom, DiningRoom, ChildRoom,
StudyRoom, SecondRoom, GuestRoom, Balcony, Entrance, Storage, Wall-in: no hallway or corridor. "Entrance" is the only
circulation class and is already mapped to `hallway`.

| | plans with an "Entrance" room |
|---|---|
| RPLAN train (20,000 checked) | 0.4% |
| RPLAN test (12,063) | 0.3% |
| our 1,000 inputs (as `hallway`) | 0.2% (2 plans) |

In RPLAN the living room is the hall: the front door opens into it in 996 of the 1,000 inputs, 87% of all interior
door connections touch it, and corridor space is drawn as part of it. In 42% of the inputs every room opens to the
living room.

### APM success rate vs. this structure (seed 0, one home per input)

| input | ≤6 rooms | 7-8 rooms | living room with <6 doors | living room with ≥6 doors |
|---|---|---|---|---|
| bubble (`next-rplan-apm-90c2ec68e2e8-bubble`, 788/1000) | 92% | 70% | 83% | 70% |
| bubble+areas (`next-rplan-apm-3ab127258fb5-bubble-areas`, 663/1000) | 82% | 56% | 70% | 57% |

### Suggestion

Adding a hallway to the inputs would give APM a different input from all other methods and would be scored as a
Types/Graph error (the comparison is against the RPLAN input). The fix belongs in APM: let the living room act as the
circulation hub. Mason2 living-room nodes already carry a `circulation` attribute.

## Current Table 1(a) rows of APM

| input | N | Valid | Reach | Types | Graph | Area | ΔFID | ΔKID |
|---|---|---|---|---|---|---|---|---|
| bubble | 788 | 86.8 | 100 | 100 | 100 | n/a | 27.6 | 31.9 |
| bubble+areas | 663 | 92.8 | 100 | 100 | 100 | 0.0% | 24.9 | 27.3 |

Real RPLAN: Valid 84.1, Reach 92.1. Full table: `results/rplan/block_a.md`. The `-retry10` versions (best of up to 10
seeds) are not in the table: every other method gets one sample per input.
