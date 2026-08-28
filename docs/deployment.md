# Deployment and persistent retrieval artifacts

## Runtime boundary

Production retrieval uses an immutable, versioned bundle. Corpus construction and BM25 indexing
are deliberate preparation work and never occur in a request:

```text
pinned SG-LegalCite + temporal split + corpus config
    -> explicit artifact build
    -> canonical manifest + gzip JSONL corpus and NumPy BM25 state
    -> read-only runtime mount
    -> verify file and payload digests
    -> restore BM25 without tokenization
    -> retrieval ready
```

The first deployment pattern is a read-only artifact volume (Option B). The application image
contains code and locked Python dependencies, but no dataset, retrieval bundle, private notes, or
credentials. This keeps the licensed data lifecycle separate from the image and permits one image
to be paired with a specifically identified bundle.

## Build the production bundle

Prepare the pinned raw dataset and temporal split as described in the main README, then run:

```bash
uv sync --locked --extra generation --extra api
uv run sg-legal-build-retrieval-artifacts
```

The default output is `data/processed/retrieval-artifacts/`. A different destination and a
reproducible timestamp can be supplied explicitly:

```bash
SOURCE_DATE_EPOCH=1700000000 uv run sg-legal-build-retrieval-artifacts \
  --output /srv/precedent/retrieval-artifacts
```

`SOURCE_DATE_EPOCH` defaults to `0`, following reproducible-build convention. It affects only the
manifest's `created_at`; it is not a claim about when the upstream dataset was published. A build
is staged in a sibling directory, loaded and validated, and then published. Rebuilding replaces an
existing destination through a staged directory swap.

The command verifies the raw CSV against `configs/dataset_manifest.toml`, constructs the
leakage-safe historical passages, builds canonical BM25, serializes both components, validates the
finished bundle, and prints document/case counts, elapsed time, and the manifest digest. It makes
no model or provider request.

## Bundle contract

```text
retrieval-artifacts/
├── manifest.json
├── corpus.jsonl.gz
├── bm25-terms.txt.gz
├── bm25-document-lengths.npy
├── bm25-idf.npy
├── bm25-posting-offsets.npy
├── bm25-posting-document-ids.npy
└── bm25-posting-frequencies.npy
```

The canonical JSON manifest records:

- artifact schema and semantic artifact versions;
- reproducible `created_at` value;
- pinned dataset ID and revision;
- source file name, size, and SHA-256;
- split and corpus-config SHA-256 values;
- corpus format, case/document counts, file size, file SHA-256, and uncompressed payload SHA-256;
- BM25 format, `k1`, `b`, document count, file size, file SHA-256, and uncompressed payload SHA-256;
- tokenization and retrieval-implementation versions.

The corpus file contains exact bounded passages, case IDs/names, source provenance, and passage
digests. The BM25 files contain a sorted term table plus fixed-width document-length, IDF, posting-
offset, document-ID, and frequency arrays. Candidate-filtered benchmark state is deliberately
excluded because production scores the complete corpus. JSON and NumPy floating-point round trips
preserve canonical Python scores; regression tests
require exact rankings, case/evidence IDs, and zero-tolerance score equivalence.

At startup the service reads only this bundle. It checks the canonical manifest, compatibility
versions, compressed-file digests, decompressed-payload digests, exact-passage digests, document
alignment, BM25 parameters, and BM25 structural invariants. Missing, corrupt, or incompatible
state is never rebuilt. The process remains live (`/health` returns 200), while `/ready` returns
503 with `status: not_ready` and `/retrieve` returns a safe 503 error. Client responses do not
contain artifact paths or validation internals.

`GET /version` exposes the artifact semantic version, manifest digest, document count, and load
time when retrieval is available. It does not expose a filesystem path.

## Local service

After building the default bundle:

```bash
uv run uvicorn sg_legal_rag.api.app:app --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/version
curl --fail http://127.0.0.1:8000/metrics
```

For a non-default bundle, set `PRECEDENT_RETRIEVAL_ARTIFACTS`. Startup, health, readiness, version,
and retrieval do not call OpenAI. Leave `OPENAI_API_KEY` unset for retrieval-only operation.

## Container build and run

```bash
docker build --tag precedent-sg-rag:local .
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --mount type=bind,src="$(pwd)/data/processed/retrieval-artifacts",dst=/opt/precedent/retrieval-artifacts,readonly \
  --publish 8000:8000 \
  precedent-sg-rag:local
```

Or use the equivalent single-service Compose file:

```bash
docker compose up --build
```

To enable `/answer`, inject a credential only at runtime; never add it to a Dockerfile, build
argument, committed environment file, or image layer:

```bash
docker run --rm \
  --env OPENAI_API_KEY \
  --mount type=bind,src="$(pwd)/data/processed/retrieval-artifacts",dst=/opt/precedent/retrieval-artifacts,readonly \
  --publish 8000:8000 \
  precedent-sg-rag:local
```

The image uses the checked-in `uv.lock`, a non-editable production environment, one Uvicorn worker
without reload, numeric non-root UID/GID `10001`, no baked secret, and a liveness-only Docker
healthcheck. `.dockerignore` excludes Git data, private `.local` material, environment files,
tests, results, reports, caches, raw data, and local artifacts. Compose additionally drops Linux
capabilities, sets `no-new-privileges`, and makes the container root filesystem read-only.

## Verification and measurements

Fixture-level build, corruption, incompatibility, no-rebuild, retrieval-equivalence, and API tests
run in the normal Python suite. CI separately builds the image, checks its configured user, mounts
a deterministic synthetic bundle, and exercises `/health`, `/ready`, `/metrics`, and `/retrieve`
with no API credential. The metrics scrape requires no writable container state.

Measured on the development host on 27 August 2026, using the complete pinned corpus:

| Measurement | Result |
| --- | ---: |
| Historical corpus construction (old first-request work) | 30.769 s |
| Canonical BM25 construction (old first-request work) | 15.928 s |
| Before: combined request-time reconstruction | 46.697 s |
| Explicit source verification + build + serialize + self-validation | 71.728 s |
| Full reproducibility rebuild | 99.975 s; identical manifest digest |
| Prepared bundle size | 188 MiB allocated (187.2 MiB file bytes) |
| Standalone verified artifact load | 2.478 s |
| After: cold process to `/health` | 2.912 s |
| After: cold process to retrieval-ready | 2.913 s |
| Retrieval-ready service maximum RSS | 720,428 KiB (about 704 MiB) |
| Example top-3 `/retrieve` HTTP latency | 37.7 ms |

The old reconstruction values are the directly timed corpus and index stages of the same explicit
production build—not an estimate from dataset size. Prepared loading is about 18.8 times faster
than the removed first-request reconstruction, and subsequent requests do no reconstruction.
The full build was repeated after adding the production equivalence gate: all six deterministic
full-corpus score probes matched canonical BM25 exactly, and both builds produced manifest digest
`e75c440618f759524745623ed87238b3e2ca8f2fbba4730cc832772e3879244d`.

Docker image size, container RSS, and container time-to-ready could not be measured on this host:
`docker`, Podman, Buildah, and nerdctl are absent. The checked-in CI job is therefore the executable
container gate. It builds the image, checks UID/GID `10001:10001`, starts it read-only with a mounted
fixture bundle, and calls `/health`, `/ready`, and `/retrieve`. Those container results remain
unverified until the workflow runs; this document does not treat Dockerfile inspection as an
executed build.

## Operational limits

Artifacts are immutable during serving but are still deployed as local files rather than through
an artifact registry. Evidence IDs remain package-local. A persistent production evidence store,
authentication, formal monitoring, multi-worker memory analysis, and rollout/rollback automation
remain future work.
