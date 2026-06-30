# Design Decisions

Locked design decisions for `lighthill`. Each entry is durable: code and specs
point here (e.g. `buoyancy_wrench` references D1). Add new decisions as D2, D3, …;
do not rewrite a locked decision — supersede it with a new entry that links back.

---

## D1 — Neutrality is expressed by volume, never by a flag

**Date:** 2026-06-29
**Status:** Locked
**Affects:** `forces.buoyancy_wrench`, `config.LinkConfig`,
`coefficients.ResolvedCoefficients`, shipped configs, design spec.

**Decision.** Buoyancy is **always** applied: `F_b = ρ·g·V` at the center of
buoyancy, contributing both force and the moment `cob_body × F_body`. A link is
made neutrally buoyant by **tuning its `volume`** so that `V = m/ρ` (buoyancy
cancels weight in net force) — the CoB↔CoM offset then still produces the
restoring couple. There is **one** way to express neutrality (volume), not two.

**What was removed.** The `neutrally_buoyant: bool` config field and the
`torch.where(neutrally_buoyant, 0, f_body)` branch in `buoyancy_wrench`. The
schema comment "skip buoyancy if true" was the originating artifact and was
corrected.

**Why the flag was wrong (a bug, not a simplification).**
- The field was borrowed from UUV-Sim's schema, but UUV-Sim's `neutrally_buoyant`
  does **not** skip the force — it derives `V = m/ρ` and applies the full
  restoring force at the CoB. lighthill implemented the literal-but-inverted
  reading of its own comment.
- Zeroing the force also zeroed the moment (`moment = cob × f`), so a neutrally
  buoyant link with an offset CoB silently lost its restoring couple
  `(r_cob − r_cog) × F_b` — exactly the term the paper's vehicle↔arm reaction
  coupling depends on.
- Under the Plan B design (PhysX applies gravity at the CoM; lighthill applies
  only buoyancy), skipping buoyancy is **doubly** wrong: it drops the couple AND
  leaves PhysX gravity uncancelled, so a "neutral" link is actually negatively
  buoyant and sinks.
- The flag had no correct behavior under any current design and was inert in
  Plan A's validated reference (which hardcoded the flag off). Nothing to preserve.

**Why delete rather than redefine (YAGNI).** A redefine-to-`V = m/ρ` helper needs
per-link mass the kernel never sees, a derivation that isn't designed, and a home
that isn't decided. The spec already declares the real convention (tune volumes
so links are ~neutral). Deleting the flag also dissolves the Plan B mass-access
dependency. If an auto-neutral convenience is genuinely needed later, build it in
Plan B where mass *is* available from the articulation, and design it properly then.

**Coverage.** `test_buoyancy_offset_cob_produces_exact_restoring_couple` locks the
previously-uncovered case: an offset CoB yields the exact couple `cob × F_b`. The
old `test_neutrally_buoyant_link_contributes_nothing` (which only passed because
it used a zero CoB offset) was deleted.
