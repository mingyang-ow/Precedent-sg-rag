# Data provenance

SG-LegalCite is downloaded from the authors' Hugging Face repository at the immutable revision
recorded in `configs/dataset_manifest.toml`. The dataset card identifies the dataset licence as
CC BY 4.0; the upstream GitHub repository identifies its code licence as MIT.

The live release inspected on 24 August 2026 contains a CSV and three benchmark JSON artifacts.
This differs from the JSONL layout currently described in the upstream GitHub `dataset/README.md`.
This project therefore validates actual artifacts and pins their revision, size, and available
SHA-256 digest rather than relying on the stale description.

## Download

```bash
uv run sg-legal-download
uv run sg-legal-download --include-benchmark
uv run sg-legal-validate-benchmark --output reports/benchmark_validation.json
```

Files are written to `data/raw/` atomically and are not committed. The downloader verifies sizes
and all published SHA-256 digests.

## Attribution

When publishing results or derived material, retain the SG-LegalCite attribution and cite the
authors' paper:

> Shannon Lee Yueh Ern, Kaidong Feng, Yingpeng Du, Chloe Lee En Jia, and Zhu Sun.
> “SG-LegalCite: A Principle-Augmented Benchmark for Legal Citation Retrieval in Singapore Law.”
> arXiv:2605.21057, 2026.

Do not bulk redistribute separately scraped judgment text. The dataset licence does not establish
permission to redistribute unrelated copies acquired from other sites.
