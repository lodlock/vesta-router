# Releases

One directory per promoted build:

```
releases/<version>/
├── manifest.json      committed
└── eval-report.json   committed
```

**The `.cact` is not here.** It is a GitHub Release asset, and it is gitignored
so it cannot be committed by accident. What is committed is the *record* of a
build — small, auditable, and outliving the weights it describes.

## Retention

| Status | Artifact | Manifest + report |
|---|---|---|
| `stable` | kept indefinitely | kept |
| `candidate` (`-rc.N`) | the promoted one and the one before it | kept |
| `experimental` (`-exp.N`) | only while its branch is open | kept |
| `rejected` | deleted | **kept** — the reason a build failed is the useful part |
| `superseded` | kept while any device may still be running it | kept |

Not every experimental `.cact` is retained. They are disposable by construction;
their manifests and eval reports are not.

## Publishing

1. Build the final `.cact` and hash it.
2. Write `manifest.json` with that digest, the corpus and config revisions, and
   the evaluation summary from the report.
3. `python -m vesta_router manifest releases/<version>/manifest.json`
4. `python -m vesta_router verify releases/<version>/manifest.json <artifact>`
5. `python -m vesta_router gate releases/<version>/eval-report.json`
6. Tag `v<version>`, create the GitHub Release, attach the `.cact` and the
   manifest.
7. Regenerate `index.json` and publish it.

Step 5 is the gate. A build that fails it is marked `rejected` and is not
released — and the gate that blocked it is named in the report, so the next
attempt knows what it is fixing.

**The evaluation must be of the artifact being released.** A `stable` manifest
whose `evaluation.evaluatedArtifactSha256` differs from its `artifact.sha256` is
refused here and on the device. Quantization changes behaviour; a number measured
before it is a number about a different file.

## The update index

`index.json` at the repository root is what Vesta fetches when the user taps
*Check for update*, and it is the only thing fetched until they tap Download. It
lists every release with its version, status, URLs, digest, size, publish date,
notes and a copy of its compatibility fields.

The copies let the app filter a list without downloading a manifest per release.
They are a **hint**: the manifest that arrives with the artifact is re-validated
from scratch on the device and wins any disagreement, because the index is the
file a stale CDN or a hostile mirror has the easiest time serving.

A `superseded` release stays listed. A device still running it needs to be able
to identify what it has.

## Hosting

GitHub Releases, for now. Versioned, immutable enough, and free to pair with a
manifest and a digest.

Vesta's side treats the URLs as data from the index rather than as a compiled-in
host, so moving to a plain static bucket later is a change to what `index.json`
contains and not a change to the app. Nothing in the app calls the GitHub API.
