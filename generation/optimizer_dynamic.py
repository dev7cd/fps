# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
generation/optimizer_dynamic.py
Densification by force-biased collective rearrangement.

Constructive placement (phase 1) is a sequential adsorption process: it rejects
any fiber that does not fit, so it saturates well below the volume fractions
needed for composite RVEs. This module goes past that limit the standard way
(Altendorf & Jeulin, 2011): insert the missing fibers with overlaps allowed,
then push every overlapping pair apart until the configuration is admissible
again, bending the fibers as they slide around each other.

The whole relaxation is periodic: a fiber is always measured against the closest
periodic image of its neighbours, and against its own images.
"""

import numpy as np
import logging
from typing import List, Tuple

from generation.periodicity import PeriodicManager
from collision.detector_math import closest_approach_periodic

## @var logger
#  @brief Logger for the dynamic optimisation module.
logger = logging.getLogger(__name__)


class DynamicOptimizer:
    """!
    @class DynamicOptimizer
    @brief Densifies an RVE past the sequential-adsorption limit.

    Alternates two steps until the target volume fraction is reached:
      1. insertion of a batch of fibers, overlaps allowed;
      2. force-biased relaxation that removes every overlap, honouring the
         configured minimum clearance and the periodic boundaries.

    A round that cannot be relaxed is rolled back, so the returned configuration
    is always free of intersections.
    """

    def __init__(self, config, detector):
        """!
        @brief Initialises the optimiser with the configuration and the collision detector.
        @param config FiberPackingConfig Problem configuration.
        @param detector CollisionDetector Collision detector.
        """
        ## @var config
        #  @brief Global configuration.
        self.config = config
        ## @var detector
        #  @brief Collision detector (owns the spatial index).
        self.detector = detector
        ## @var rng
        #  @brief Random number generator.
        self.rng = np.random.default_rng(config.seed)
        ## @var dims
        #  @brief Domain dimensions as an array, for the periodic kernels.
        self.dims = np.array(config.box_dims, dtype=float)
        ## @var clearance
        #  @brief Minimum empty gap required between two fiber surfaces.
        self.clearance = getattr(detector, 'min_clearance', 0.0)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def optimize_and_fill(self, fibers: List):
        """!
        @brief Densification cycle: insert a batch, then relax the overlaps away.
        @param fibers List List of initial fibers to densify.
        @return List Updated and completed list of fibers.
        """
        target_vf = self.config.target_volume_fraction
        current_vf = self._calculate_vf(fibers)

        logger.info(f"Densification start. Current Vf: {current_vf:.4f} "
                    f"(target {target_vf:.4f})")

        if not fibers:
            logger.warning("No fiber to densify.")
            return fibers

        # Relax whatever phase 1 produced before adding anything on top.
        self._relax(fibers)

        mean_vol = np.mean([f.get_real_volume() for f in fibers])
        box_vol = float(np.prod(self.config.box_dims))
        max_rounds = max(1, self.config.optimizer_iterations)
        # Halved after every failed round: near saturation, asking again for the
        # same batch would just fail the same way and burn a full relaxation.
        batch_cap = max(1, self.config.injection_per_iteration)

        for round_idx in range(1, max_rounds + 1):
            current_vf = self._calculate_vf(fibers)
            if current_vf >= target_vf:
                break

            missing = (target_vf - current_vf) * box_vol / mean_vol
            batch_size = int(np.clip(np.ceil(missing), 1, batch_cap))

            batch = self._spawn_batch(batch_size, start_id=self._next_id(fibers))
            if not batch:
                logger.warning("Insertion produced no fiber; densification stops.")
                break

            snapshot = {id(f): f.control_points.copy() for f in fibers}
            fibers.extend(batch)

            resolved, residual = self._relax(fibers)
            if resolved:
                logger.info(f"Round {round_idx}: +{len(batch)} fibers, "
                            f"Vf = {self._calculate_vf(fibers):.4f}")
                continue

            # The batch could not be accommodated: drop the newcomers that are
            # still in contact, restore the others to their pre-round shape,
            # and let the next round try a smaller batch.
            batch_cap = max(1, batch_size // 2)
            fibers[:] = self._drop_offenders(fibers, batch, residual)
            for f in fibers:
                pts = snapshot.get(id(f))
                if pts is not None:
                    f.control_points = pts
                    f.refresh_geometry()

            recovered, _ = self._relax(fibers)
            if not recovered:
                # Reverting the old fibers can re-create contacts with the
                # newcomers that were kept. Nothing from this round survives.
                fibers[:] = [f for f in fibers if id(f) in snapshot]
                self._relax(fibers)
                logger.info(f"Round {round_idx}: batch rejected, "
                            f"Vf = {self._calculate_vf(fibers):.4f}")
                break

            logger.info(f"Round {round_idx}: batch partially rejected, "
                        f"Vf = {self._calculate_vf(fibers):.4f}")

            if len(batch) <= 1:
                logger.info("Packing saturated: a single extra fiber no longer fits.")
                break

        self.detector.rebuild_index(fibers)
        final_vf = self._calculate_vf(fibers)
        if final_vf < target_vf:
            logger.warning(f"Densification stopped at Vf = {final_vf:.4f}, "
                           f"below the {target_vf:.4f} requested.")
        else:
            logger.info(f"Densification finished. Vf = {final_vf:.4f}")
        return fibers

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def _spawn_batch(self, count: int, start_id: int) -> List:
        """!
        @brief Creates a batch of fibers, overlaps allowed.
        @details The walk keeps the curvature and step-length statistics of
        phase 1 but skips the collision tests: the relaxation is what makes the
        configuration admissible afterwards.
        @param count int Number of fibers to create.
        @param start_id int First identifier to assign.
        @return List[Fiber] The created fibers.
        """
        from generation.generator import FiberGenerator

        gen = FiberGenerator(self.config, self.detector)
        gen.rng = self.rng  # keep one reproducible stream for the whole phase
        batch = []
        for k in range(count):
            fiber = gen.generate_fiber(start_id + k, check_collisions=False)
            if fiber is not None:
                PeriodicManager.wrap_fiber(fiber, self.config.box_dims)
                batch.append(fiber)
        return batch

    def _next_id(self, fibers: List) -> int:
        """!
        @brief Returns the first free fiber identifier.
        """
        return max((f.id for f in fibers), default=0) + 1

    # ------------------------------------------------------------------
    # Force-biased relaxation
    # ------------------------------------------------------------------

    def _relax(self, fibers: List) -> Tuple[bool, dict]:
        """!
        @brief Pushes every overlapping pair apart until the RVE is admissible.
        @details One sweep collects all the contacts, applies the corresponding
        displacements at once (Jacobi style, so the result does not depend on the
        fiber ordering), then restores the chain geometry of each fiber.
        @param fibers List Fibers to relax, modified in place.
        @return tuple (resolved, residual) with residual mapping a fiber index to
                its worst remaining penetration.
        """
        max_sweeps = self.config.densifier_sweeps
        damping = self.config.densifier_damping
        stall_kick = self.config.jitter_intensity * self.config.fiber_radius
        # Relaxing to a slightly larger separation than required turns an
        # asymptotic tail (overlaps shrinking towards zero forever) into a loop
        # that terminates with a genuine, positive gap.
        margin = self.config.densifier_margin * self.config.fiber_radius

        previous = np.inf
        stalled = 0
        cache = {}
        dirty = set(range(len(fibers)))

        resolved = False
        for sweep in range(max_sweeps):
            contacts = self._collect_contacts(fibers, margin, cache, dirty)
            if not contacts:
                resolved = True
                break

            worst = max(c[4] for c in contacts)
            moved = self._apply_pushes(fibers, contacts, damping)

            # Only the fibers actually pushed need their geometry recomputed,
            # and the Bishop frames are not used by the collision tests.
            for index in moved:
                fiber = fibers[index]
                self._restore_chain(fiber)
                fiber.refresh_geometry(with_frames=False)

            # A configuration that stops improving gets a random kick: it is
            # cheaper than restarting and usually unlocks the jammed pair.
            if worst > previous - 1e-6:
                stalled += 1
                if stalled >= 10:
                    moved |= self._kick(fibers, contacts, stall_kick)
                    stalled = 0
            else:
                stalled = 0
            previous = worst
            dirty = moved

            if sweep % 50 == 0:
                logger.debug(f"  sweep {sweep}: {len(contacts)} contacts, "
                             f"worst penetration {worst:.2e}")

        # Back into the box, frames included: what leaves this method is meshed
        # and voxelised downstream.
        for fiber in fibers:
            PeriodicManager.wrap_fiber(fiber, self.config.box_dims)
            fiber.refresh_geometry()

        if resolved:
            return True, {}

        residual = {}
        for i, j, _u1, _u2, penetration, _direction in self._collect_contacts(fibers):
            residual[i] = max(residual.get(i, 0.0), penetration)
            residual[j] = max(residual.get(j, 0.0), penetration)
        return (not residual), residual

    def _collect_contacts(self, fibers: List, margin: float = 0.0,
                          cache: dict = None, dirty: set = None) -> List[Tuple]:
        """!
        @brief Lists every pair closer than the required separation.
        @details A pair whose two fibers both stood still since the previous
        sweep cannot have changed, so its result is reused from the cache. Late
        in a relaxation only a handful of fibers still move and the scan becomes
        much cheaper than the quadratic worst case.
        @param fibers List Fibers to scan (indices are used as identifiers).
        @param margin float Extra separation demanded on top of radii and clearance.
        @param cache dict Optional (i, j) -> contact or None, updated in place.
        @param dirty set Optional indices whose geometry changed since last call.
        @return List of tuples (i, j, u_i, u_j, penetration, direction).
        """
        n = len(fibers)
        contacts = []
        reuse = cache is not None and dirty is not None

        # Bounding boxes padded by the interaction range, for a cheap rejection.
        reach = 2 * self.config.fiber_radius + self.clearance + margin
        boxes = [(f.bbox[0] - reach, f.bbox[1] + reach) for f in fibers]

        for i in range(n):
            fi = fibers[i]
            for j in range(i, n):
                if reuse and i not in dirty and j not in dirty and (i, j) in cache:
                    hit = cache[(i, j)]
                    if hit is not None:
                        contacts.append(hit)
                    continue

                fj = fibers[j]
                required = fi.radius + fj.radius + self.clearance + margin
                if i != j and not self._boxes_may_touch(boxes[i], boxes[j]):
                    hit = None
                else:
                    # i == j measures the fiber against its own periodic images
                    d, u1, u2, ux, uy, uz = closest_approach_periodic(
                        fi.centerline, fj.centerline, self.dims, i == j, required)
                    hit = None
                    if d < required:
                        hit = (i, j, u1, u2, required - d, np.array([ux, uy, uz]))

                if reuse:
                    cache[(i, j)] = hit
                if hit is not None:
                    contacts.append(hit)
        return contacts

    def _boxes_may_touch(self, box_a, box_b) -> bool:
        """!
        @brief Periodic bounding-box rejection test.
        @details Boxes are already padded by the interaction range. Two fibers
        can interact through any face, so the test is run on the separation
        folded back into the domain.
        """
        min_a, max_a = box_a
        min_b, max_b = box_b
        centre_a = 0.5 * (min_a + max_a)
        centre_b = 0.5 * (min_b + max_b)
        half = 0.5 * ((max_a - min_a) + (max_b - min_b))
        delta = np.abs(centre_a - centre_b)
        # Fold each axis: a separation larger than half a period is shorter the
        # other way round. Fibers wider than the box always pass the test.
        folded = np.minimum(delta, np.maximum(self.dims - delta, 0.0))
        return bool(np.all(folded <= half))

    def _apply_pushes(self, fibers: List, contacts: List[Tuple], damping: float):
        """!
        @brief Applies every contact displacement at once.
        @details Each contact moves the two fibers apart by half the penetration,
        spread over the control points around the contact with a Gaussian weight:
        the fiber bends locally instead of translating as a rigid body, which is
        what lets the packing go past the adsorption limit.
        @param fibers List Fibers being relaxed.
        @param contacts List Contacts returned by _collect_contacts.
        @param damping float Fraction of the penetration corrected per sweep.
        @return set Indices of the fibers that were displaced.
        """
        deltas = {}
        for i, j, u1, u2, penetration, direction in contacts:
            if np.allclose(direction, 0.0):
                direction = self.rng.standard_normal(3)
                direction /= np.linalg.norm(direction)

            amplitude = 0.5 * damping * penetration
            for index, abscissa, sign in ((i, u1, 1.0), (j, u2, -1.0)):
                push = sign * amplitude * self._weights(fibers[index], abscissa)[:, None] * direction
                if index in deltas:
                    deltas[index] += push
                else:
                    deltas[index] = push

        for index, delta in deltas.items():
            fibers[index].control_points = fibers[index].control_points + delta
        return set(deltas)

    def _weights(self, fiber, u_centerline: float) -> np.ndarray:
        """!
        @brief Gaussian weights over the control points around a contact.
        @param fiber Fiber The fiber being pushed.
        @param u_centerline float Contact abscissa, as a fractional centerline index.
        @return np.ndarray Weights peaking at 1.0 on the closest control point.
        """
        n_cp = len(fiber.control_points)
        n_seg = max(1, len(fiber.centerline) - 1)
        # Map the centerline abscissa onto the control polygon
        position = u_centerline / n_seg * (n_cp - 1)
        idx = np.arange(n_cp, dtype=float)
        sigma = 2.0
        return np.exp(-0.5 * ((idx - position) / sigma) ** 2)

    def _kick(self, fibers: List, contacts: List[Tuple], amplitude: float):
        """!
        @brief Random displacement of the fibers still in contact.
        @details Used when the relaxation stops making progress, to break the
        symmetry of a jammed pair.
        @return set Indices of the fibers that were kicked.
        """
        involved = set()
        for i, j, *_ in contacts:
            involved.add(i)
            involved.add(j)
        for i in involved:
            fibers[i].control_points = fibers[i].control_points + self.rng.normal(0.0, amplitude, 3)
            fibers[i].refresh_geometry(with_frames=False)
        return involved

    # ------------------------------------------------------------------
    # Chain geometry
    # ------------------------------------------------------------------

    def _restore_chain(self, fiber):
        """!
        @brief Restores the step length and curvature of a pushed fiber.
        @details The contact displacements stretch and kink the control polygon.
        Segment lengths are pulled back to the generation step (position-based
        projection), then the points whose turn exceeds the curvature limit are
        smoothed. Without this the relaxation would silently change the fiber
        length and tortuosity statistics.
        @param fiber Fiber The fiber to fix up, modified in place.
        """
        pts = fiber.control_points
        n = len(pts)
        if n < 2:
            return

        target_step = self.config.step_length_mean
        # Phase 1 advances by exactly one step per control point, so the chain is
        # rebuilt with that spacing: keep the edge directions the contacts just
        # produced, reset their lengths, and re-centre so the fiber does not
        # drift away from where it was pushed.
        edges = pts[1:] - pts[:-1]
        lengths = np.linalg.norm(edges, axis=1)
        directions = np.empty_like(edges)
        previous = np.array([1.0, 0.0, 0.0])
        for k in range(len(edges)):
            if lengths[k] > 1e-12:
                previous = edges[k] / lengths[k]
            directions[k] = previous

        centroid = pts.mean(axis=0)
        rebuilt = np.empty_like(pts)
        rebuilt[0] = pts[0]
        rebuilt[1:] = pts[0] + np.cumsum(target_step * directions, axis=0)
        pts = rebuilt + (centroid - rebuilt.mean(axis=0))

        max_angle = self.config.generation_parameters.get('max_curvature_angle', np.pi / 6)
        smoothing = self.config.densifier_smoothing
        if smoothing > 0.0 and n >= 3:
            cos_limit = np.cos(max_angle)
            for k in range(1, n - 1):
                v1 = pts[k] - pts[k - 1]
                v2 = pts[k + 1] - pts[k]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-12 or n2 < 1e-12:
                    continue
                if np.dot(v1, v2) / (n1 * n2) < cos_limit:
                    midpoint = 0.5 * (pts[k - 1] + pts[k + 1])
                    pts[k] = pts[k] + smoothing * (midpoint - pts[k])

        fiber.control_points = pts

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _drop_offenders(self, fibers: List, batch: List, residual: dict) -> List:
        """!
        @brief Removes the newly inserted fibers that are still in contact.
        @param fibers List Full fiber list.
        @param batch List Fibers inserted during the current round.
        @param residual dict Fiber index -> worst remaining penetration.
        @return List The fibers to keep.
        """
        batch_ids = {id(f) for f in batch}
        offenders = {i for i, pen in residual.items() if pen > 0.0}
        keep = []
        for index, fiber in enumerate(fibers):
            if index in offenders and id(fiber) in batch_ids:
                continue
            keep.append(fiber)
        return keep

    def _calculate_vf(self, fibers):
        """!
        @brief Computes the current real volume fraction.
        """
        if not fibers:
            return 0.0
        total_vol = sum(f.get_real_volume() for f in fibers)
        return total_vol / np.prod(self.config.box_dims)
