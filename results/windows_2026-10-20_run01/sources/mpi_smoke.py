"""Check MPI rank conventions, NumPy Scatter buffers, and Python gather objects."""

import json
import sys

import numpy as np
from mpi4py import MPI


def main() -> int:
    """Scatter two integers per rank, gather their sums, and verify on root 0."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    try:
        # Only root owns the complete input. Uppercase Scatter moves array data.
        send_buffer = None
        if rank == root:
            send_buffer = np.arange(size * 2, dtype=np.int64).reshape(size, 2)
        received = np.empty(2, dtype=np.int64)
        comm.Scatter(send_buffer, received, root=root)

        # Lowercase gather serializes ordinary Python objects, returning a list
        # in rank order on root and None on the other ranks.
        record = {
            "rank": rank,
            "size": size,
            "received": received.tolist(),
            "sum": int(received.sum()),
        }
        gathered = comm.gather(record, root=root)

        exit_code = 0
        if rank == root:
            errors = []
            if [item["rank"] for item in gathered] != list(range(size)):
                errors.append("Gathered ranks do not match communicator order.")
            for expected_rank, item in enumerate(gathered):
                expected_values = [2 * expected_rank, 2 * expected_rank + 1]
                if item["size"] != size:
                    errors.append(f"Rank {expected_rank}: incorrect communicator size.")
                if item["received"] != expected_values:
                    errors.append(f"Rank {expected_rank}: incorrect Scatter values.")
                if item["sum"] != sum(expected_values):
                    errors.append(f"Rank {expected_rank}: incorrect computed sum.")
            exit_code = int(bool(errors))
            print(
                json.dumps(
                    {
                        "status": "FAIL" if errors else "PASS",
                        "root": root,
                        "size": size,
                        "gathered": gathered,
                        "errors": errors,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # All processes return the same status after root has checked results.
        return comm.bcast(exit_code, root=root)
    except Exception as error:
        # A rank failing before a collective could otherwise leave peers waiting.
        print(f"Rank {rank}: MPI smoke test failed: {error}", file=sys.stderr, flush=True)
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
