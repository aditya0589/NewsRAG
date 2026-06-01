# Building the FAISS Lambda Layer

The FAISS library has C++ binaries that must match the Lambda execution
environment exactly. This means you cannot simply `pip install faiss-cpu`
on your local machine and zip it up — the binaries will be built for your
OS, not for Amazon Linux 2023 (which Lambda runs on).

The solution is to build inside a Docker container that matches the Lambda
runtime. This guide covers Windows, macOS, and Linux.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- ~500MB free disk space for the build

Verify Docker is running before proceeding:

```bash
docker info
```

If this errors, start Docker Desktop and wait for the whale icon in your
system tray to stop animating.

---

## Build instructions

### Windows (Command Prompt)

Run these commands one at a time from any working directory.

**Step 1 — Create a build folder and enter it:**
```cmd
mkdir faiss-layer
cd faiss-layer
```

**Step 2 — Build inside the Lambda Docker image:**
```cmd
docker run --rm --entrypoint pip -v %cd%:/out public.ecr.aws/lambda/python:3.11 install faiss-cpu==1.7.4 numpy==1.26.4 packaging -t /out/python
```

**Step 3 — Zip the output:**
```cmd
powershell -Command "Compress-Archive -Path python -DestinationPath faiss-layer.zip"
```

---

### macOS / Linux (Terminal)

**Step 1 — Create a build folder and enter it:**
```bash
mkdir faiss-layer && cd faiss-layer
```

**Step 2 — Build inside the Lambda Docker image:**
```bash
docker run --rm --entrypoint pip \
  -v $(pwd):/out \
  public.ecr.aws/lambda/python:3.11 \
  install faiss-cpu==1.7.4 numpy==1.26.4 packaging -t /out/python
```

**Step 3 — Zip the output:**
```bash
zip -r faiss-layer.zip python/
```

---

## Why these specific versions?

| Package | Version | Reason |
|---|---|---|
| `faiss-cpu` | `1.7.4` | Last version with a pre-built Linux wheel for Python 3.11. Newer versions (1.8+) attempt to build from source inside the container, which fails because the image lacks a C compiler. |
| `numpy` | `1.26.4` | ABI-compatible with faiss-cpu 1.7.4. NumPy 2.x breaks the C extension interface faiss relies on. |
| `packaging` | latest | Required by faiss-cpu at import time. Omitting it causes `No module named 'packaging'` at Lambda cold start. |

---

## Uploading to AWS Lambda

1. Go to **AWS Console → Lambda → Layers → Create layer**
2. Fill in the form:

| Field | Value |
|---|---|
| Name | `faiss-layer` |
| Upload | select `faiss-layer.zip` |
| Compatible runtimes | Python 3.11 |
| Compatible architectures | x86_64 |

3. Click **Create**
4. Copy the **Layer ARN** shown after creation — you'll need this when attaching to Lambda functions

---

## Attaching the layer to a Lambda function

1. Open your Lambda function in the console
2. Go to the **Code** tab
3. Scroll down to the **Layers** section
4. Click **Add a layer**
5. Select **Specify an ARN** and paste the layer ARN
6. Click **Add**

Repeat for both `rag-ingest` and `rag-query`.

---

## Rebuilding after changes

If you need to update the layer (e.g. to add a new package):

1. Delete the existing `python/` folder:
   - Windows: `rmdir /s /q python`
   - macOS/Linux: `rm -rf python`

2. Delete the old zip:
   - Windows: `del faiss-layer.zip`
   - macOS/Linux: `rm faiss-layer.zip`

3. Re-run the Docker build command with your updated packages

4. Re-zip and upload as a **new layer version** in the console
   - Lambda → Layers → `faiss-layer` → Create version
   - After uploading, update each Lambda function to point to the new version ARN

---

## Troubleshooting

**`entrypoint requires the handler name to be the first argument`**

You forgot `--entrypoint pip`. The Lambda Docker image expects a handler
function name as its default entrypoint. Always include `--entrypoint pip`
before the image name.

**`error: metadata-generation-failed` during build**

You're installing a version of faiss-cpu or numpy that tries to compile
from source. Pin to the exact versions specified above:
`faiss-cpu==1.7.4 numpy==1.26.4`

**`No module named 'faiss'` at Lambda runtime**

The layer is not attached to the function, or you attached the wrong
version ARN. Go to Lambda → your function → Code tab → Layers and verify
the faiss-layer ARN is listed.

**`No module named 'packaging'` at Lambda runtime**

Rebuild the layer including `packaging` in the install command.
The build command above already includes it.

**Layer zip is too large (>50MB unzipped limit warning)**

The unzipped layer must be under 250MB. faiss-cpu + numpy + packaging
comes to ~120MB unzipped, well within limits. If you add other packages
and hit the limit, consider splitting into a separate numpy layer.

**`The system cannot find the file specified` on Windows**

Docker Desktop is not running. Open Docker Desktop from the Start menu
and wait for it to fully load before re-running the command.