# Deployment

How to put this project in front of researchers so they can fill a form, press
**Run**, and get a 3D model back — and how to run the same thing from a terminal
or CI.

---

## 1. Recommendation

**Free Docker web app on Render (web form + API) + the Kaggle API for GPU
execution + the Hugging Face Hub as storage.**

| Option | Cost | GPU | Operational burden | Verdict |
| --- | --- | --- | --- | --- |
| **Render free web service → Kaggle API** — this repo | $0 (sleeps when idle) | in the researcher's own Kaggle account (their quota) | low: connect the repo, set two env vars | **recommended** |
| HF Space (Docker) → Kaggle API | HF **PRO** (~$9/mo) + ~0 runtime | same | low | good if you already have PRO; Free accounts get HTTP 402 |
| HF **Static** Space (viewer only) + CLI | $0 | same | researchers run `run_pipeline.py` | free fallback, no Run button |
| HF Space with a GPU upgrade (`sdk: docker` + `t4-small`…) | ~$0.6–3.5/h while running | built in | you pay while idle; must move 7 notebooks onto the Space | overkill for a batch pipeline |
| Kaggle notebook only (no web UI) | 0 | 30 h GPU/week free | researchers must edit YAML by hand | good fallback, not a product |
| A VM (GCP/AWS/EC2/SageMaker) with the full stack | ~$50+/month | optional | you own the OS, the GPU drivers, the nnU-Net install | wrong scale for this project |
| Colab + Drive | free-ish | optional | fragile: no long-lived service, session limits | no |

Why this split is the professional choice for a research pipeline:

* **the web tier must be cheap and boring** — it only validates a form, fills in
  parameters, pushes notebooks and polls two public APIs. A free Render web
  service covers it (Hugging Face now gates Docker/Gradio Spaces behind PRO;
  only Static Spaces are free, and this app needs a server).
* **the heavy work is bursty and long** — registration, hours of nnU-Net
  training and GPU inference. Kaggle gives researchers GPU hours in *their own*
  account, so the project never bills anyone for an idle GPU.
* **Hugging Face is already the transport** — every stage reads its inputs from
  and writes its outputs to the Hub, so no shared filesystem is needed between
  the form, the notebooks and the viewer. Run state is mirrored there too, so a
  free host that sleeps does not lose a multi-hour run.

```
 browser ──► Web app (FastAPI on Render free)          researcher's machine / CI
               │  form → run spec                       └─ run_pipeline.py --spec …
               │  background worker
               ▼
             Kaggle API ──► kernel per notebook (GPU when needed)
               │                  │ HF_TOKEN injected into the private kernel
               │                  ▼
               │        Hugging Face Hub
               │        volumes/<id>/ slices/<id>/ viewers/<id>/ runs/<id>/
               ▼
             GET /api/runs/<id> ──► status card ──► /viewer/?specimen=<id>
```

> Render's free plan spins down after ~15 min without traffic and its disk is
> ephemeral, so the app mirrors `state.json` to `runs/<id>/` on the Hub and
> auto-resumes (adopting any in-flight Kaggle kernel instead of re-running it).
> Keep the status tab open during a run to keep the worker awake.

---

## 2. One-time setup (≈15 minutes)

### a. Own the Hugging Face repositories

The deployment uses three project repositories:

| Purpose | Default | Field in the spec / form |
| --- | --- | --- |
| shared outputs (`volumes/`, `slices/`, `viewers/`, run state, uploads) | `senku111/embryo-3d` | `input.hf_repo` / `PFA_SHARED_REPO` |
| CS17 groups manifest + training dataset (case 1) | `senku111/atlas-cs17-4groups-dataset` | `models.dataset_repo` |
| CS17 4-group checkpoints (case 1 only) | `senku111/atlas-cs17-4groups-nnunet-training` | `models.training_repo` |

The training dataset and checkpoint repos are **not needed by cases 2 and 3**:
those cases use the researcher's model repo. They remain because case 1
reproduces the project models and because the report documents the training
chain. If the dataset repo is private or missing, case 1 has a read-only
fallback: the reconstruction notebook derives the group layout and Atlas ids
from the `dataset.json` files in the checkpoint repo, then applies the CS17
normalisation range. No extra upload is needed.

A token can only write to repositories the account owns (or is invited to), so
**a researcher who wants their own specimens should point the form at their own
dataset repo** — or be added as a collaborator to the shared one. A missing
shared dataset is created by `pipeline/hfio.py` when the first artifact is
pushed. Form uploads go to `PFA_UPLOAD_REPO` (defaults to the shared dataset);
set it to a separate dataset if you do not want uploads mixed with artifacts.

`POST /api/uploads` is intentionally unauthenticated so the public form works.
Protect it with a host-level WAF/rate limit or a separate upload repo if the
deployment is public; the default size cap is `PFA_MAX_UPLOAD_MB=2048`.

### b. Hugging Face token (app secret)

1. <https://huggingface.co/settings/tokens> → **Fine-grained** token with
   *write* access to the repositories you will use.
2. Add it to the host:
   * **Render** → Environment → `HF_TOKEN`
   * **HF Space** → Settings → Variables and secrets → New secret `HF_TOKEN`

Used by the web app (`pipeline/hfio.py`) to validate runs, mirror run state and
serve the specimen picker.

### c. Kaggle API credentials (one app secret)

Kaggle issues an **access token** for new accounts:

1. <https://www.kaggle.com/settings> → **API → Create New Token**.
   * You get either a `KGAT_...` access token (new flow) or a `kaggle.json`
     containing `username` + `key` (legacy flow).
2. Add a secret:
   * `KAGGLE_API_TOKEN` = the `KGAT_...` access token (new flow), **or**
   * `KAGGLE_USERNAME` + `KAGGLE_KEY` (legacy `kaggle.json`).

The access token *contains* the username, so `KAGGLE_USERNAME` is not needed with
it — the runner introspects the account from the token when it builds the
`<user>/<slug>` kernel id. Local/CI use works the same way: the Kaggle CLI reads
`KAGGLE_API_TOKEN`, then `~/.kaggle/access_token`, then the legacy
`~/.kaggle/kaggle.json`.

The web app uses these to push, poll and download kernels. Nothing is written to
the researcher's disk.

> The Kaggle account that runs the kernels may be **different** from the Hugging
> Face account that owns the repositories — they only need to be wired together
> through the `HF_TOKEN` handling in step (d).

### d. Kaggle-side credentials

The runner **injects `HF_TOKEN` into every private kernel it pushes** (a
one-cell `os.environ` preamble). This is required because Kaggle's push API
cannot attach account secrets to a freshly created kernel. The kernel is
private, the value is never printed, and nothing about it is mirrored to the
Hub.

Kaggle Secrets are still supported as a fallback for notebooks you run by hand:
**Settings → Add-ons → Secrets → New secret**, name `HF_TOKEN`. Every notebook
reads `os.environ["HF_TOKEN"]` first and then `UserSecretsClient()`, so either
path works.

Also make sure **GPU** is enabled on that Kaggle account for the GPU stages.

### e. Deploy the web app (free)

**Option 1 — Render from GitHub (recommended):**

1. Push this repo to GitHub (the project is at
   <https://github.com/EngSenku/pfa-embryo-3d>).
2. <https://dashboard.render.com> → **New → Web Service → Public Git
   Repository**, URL: `https://github.com/<you>/pfa-embryo-3d`.
3. Runtime **Docker**, plan **Free**, nearest region, health check
   `/api/health`.
4. Add environment variables:
   * `HF_TOKEN`
   * `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`)
   * `PFA_SHARED_REPO` = `senku111/embryo-3d`
   * `PFA_RUNS_DIR` = `/data/pfa-runs`
   * optional: `PFA_UPLOAD_REPO`, `PFA_MAX_UPLOAD_MB`
5. **Create Web Service**. Render may ask for card verification: the plan stays
   **$0/month** and the $1 authorization is released, it is not a charge.
6. Wait for the first build (~3–5 min), open the URL, and confirm
   `GET /api/version` returns the current `APP_VERSION`.

After a `git push`, use **Manual Deploy → Deploy latest commit** (public-URL
services do not always auto-deploy). The included [`render.yaml`](render.yaml)
also works as a Blueprint if the repo is connected through the GitHub provider
(auto-deploy on push).

Notes: Render injects `$PORT` (the Dockerfile honours it), free services sleep
after ~15 min idle (first visit ~30–50 s), and the disk is ephemeral — run state
is mirrored to `runs/<id>/` on the Hub and auto-resumed, so this is safe.

> `senku111/pfa-source` is an older source mirror and is **not used** by the
> current deployment. Render cannot build from a Hugging Face dataset URL; use
> GitHub (or an image registry). Keep or delete that repo independently.

**Option 2 — Hugging Face Space (only if the account has PRO):**

Free HF accounts can only create *Static* Spaces; `create_repo` returns HTTP 402
`Payment Required` for a Docker Space without PRO (<https://huggingface.co/pro>).

```bash
export HF_TOKEN=hf_...
tools/deploy_space.sh senku111/pfa-embryo3d      # create + push
```

or manually:

```bash
huggingface-cli repo create pfa-embryo3d --type space --space-sdk docker
git remote add space https://huggingface.co/spaces/<you>/pfa-embryo3d
git push space main
```

The `README.md` front matter already declares `sdk: docker` and the `Dockerfile`
copies the project in. Secrets are **Settings → Variables and secrets** — the
build never sees them.

### f. Smoke test

1. Open the web app → `GET /api/health` must report `"ok": true` (fast,
   local-only check). `GET /api/health?deep=1` additionally resolves the Kaggle
   account over the network (`creds.kaggle_account`).
2. Press **Plan** with the default (case 1) form → you should see
   `registration` + `reconstruct`.
3. Switch to **case 2**, enter another Atlas specimen id, your model repo,
   configuration and fold → **Plan** shows the same two stages with
   `MODEL_SOURCE=custom`.
4. Switch to **case 3**, paste a small ZIP or choose one, enter XY/Z µm →
   **Plan** uploads the archive (`POST /api/uploads`) and shows
   `input.format=raw_section_archive` with your µm values.
5. Press **Dry run (prepare only)** → status reaches `done` in a few seconds
   without touching Kaggle (registration is skipped when the volume already
   exists).
6. Press **Run** → the first stage appears as a kernel under
   `kernels/list?search=pfa-`. The kernel slug follows the notebook, so the
   runner polls the slug Kaggle reports rather than a guessed one.
7. Open the viewer link after reconstruction; for a non-CS17 bundle, confirm the
   header uses the bundle title and the CS17 Dice card is absent.

---

## 3. Day-to-day use

| Who | What |
| --- | --- |
| Researcher (no terminal) | open the web app, pick one of the three cases, **Plan**, **Run**, follow the status card, open the viewer link |
| Researcher (own model) | train/publish an nnU-Net v2 model repo first, then use case 2 or 3 with its groups manifest, `configuration` and `fold` |
| Researcher (terminal/CI) | `python3 run_pipeline.py --spec specs/my-run.yaml [--engine kaggle] [--resume]` |
| Maintainer | read `GET /api/runs` on the web app, or `runs/<run_id>/` on disk |
| Anyone | browse `https://huggingface.co/datasets/<repo>/tree/main/viewers/<id>` |

Resuming: every run stores `state.json` locally **and** mirrors it to
`runs/<run_id>/state.json` on the Hub. `--resume` (or **Run with resume**) skips
the stages already marked `done` and adopts an in-flight Kaggle kernel rather
than pushing a duplicate. On a free host that sleeps, the next visit to
`/api/runs/<id>` resumes the run automatically.

Artifacts are verified after every stage (`volumes/<id>`, the groups manifest,
`viewers/<id>/manifest.json`), so a kernel that "succeeds" without pushing
anything fails the run with the log path.

---

## 4. Limits and knobs

* **Kaggle GPU quota** — free accounts get a limited GPU allowance per week;
  long training runs consume it and are the most common reason to use
  `--resume`. GPU is only requested for `train_group*` and `reconstruct`.
* **Kernel timeout** — `run.timeout_s` (default 6 h per stage, 24 h for
  training specs). Configurable via the form.
* **Hugging Face rate limits** — set `HF_TOKEN`; anonymous Hub calls are
  slower and eventually throttled.
* **Runs storage** — `PFA_RUNS_DIR` (defaults to `/data/pfa-runs` when that
  mount exists, `./runs` otherwise). The authoritative copy of each run's state
  and log is mirrored to `runs/<run_id>/` on the Hub, in `PFA_STATE_REPO`
  (defaults to `PFA_SHARED_REPO`).
* **Large shared datasets** — the viewer fetches from the Hub by URL; nothing
  is proxied through the web app, so the web tier stays free.
* **Form uploads** — `PFA_MAX_UPLOAD_MB` (default 2048). The archive is
  streamed to disk, uploaded to `PFA_UPLOAD_REPO`, then deleted. For anything
  larger, give a Hugging Face dataset link. The endpoint has no application
  login: put the host behind a WAF/rate limit if it is public.
* **Custom models** — `configuration` and `fold` must match the published
  checkpoint. A mismatched configuration fails in `nnUNetv2_predict`; a
  mismatched fold fails when the notebook looks for `fold_N/`.

---

## 5. Operational checklist

```
[ ] Render web service created from the public GitHub repo (Docker, Free)
[ ] Web-app env vars: HF_TOKEN + KAGGLE_API_TOKEN  (or KAGGLE_USERNAME + KAGGLE_KEY)
[ ] Optional: PFA_UPLOAD_REPO + PFA_MAX_UPLOAD_MB
[ ] Kaggle GPU enabled; HF_TOKEN is injected into each pushed private kernel
[ ] Repos exist (or the researcher's repos are filled in on the form)
[ ] After each push: Manual Deploy → Deploy latest commit
[ ] GET /api/version → current APP_VERSION and commit
[ ] GET /api/health → ok
[ ] Case 1 Atlas Plan → registration + reconstruct
[ ] Case 2 custom model Plan → MODEL_SOURCE=custom, configuration/fold present
[ ] Case 3 small ZIP upload → POST /api/uploads returns a URL, raw µm present
[ ] Dry run → done
[ ] Real run → kernels visible in Kaggle, artifacts visible on the Hub
[ ] /viewer/?specimen=<id> renders the bundle
[ ] Non-CS17 bundle: header title + no CS17 Dice card
```
