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
               │                  │ reads HF_TOKEN from Kaggle Secrets
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

The defaults point at the project's public repositories:

| Purpose | Default | Field in the spec / form |
| --- | --- | --- |
| shared outputs (`volumes/`, `slices/`, `viewers/`) | `senku111/embryo-3d` | `input.hf_repo` |
| training datasets + groups manifest | `senku111/atlas-cs17-4groups-dataset` | `models.dataset_repo` |
| training checkpoints | `senku111/atlas-cs17-4groups-nnunet-training` | `models.training_repo` |

A token can only write to repositories the account owns (or is invited to), so
**a researcher who wants their own specimens should create their own three
repositories** — or be added as a collaborator to the shared ones. The training
notebook creates a missing model repo automatically, the others are created by
`pipeline/hfio.py` when the first artifact is pushed.

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
> through the `HF_TOKEN` Kaggle secret in step (d).

### d. Kaggle-side secret

In the **same Kaggle account that will run the kernels**, add
**Settings → Add-ons → Secrets → New secret**: name `HF_TOKEN`, the Hugging Face
token above. Every notebook reads it before touching the Hub
(`os.environ.get("HF_TOKEN")` → `UserSecretsClient()`), so without it the first
push fails with `HF_TOKEN unavailable`.

Also make sure **GPU** is enabled on that Kaggle account for the GPU stages.

### e. Deploy the web app (free)

**Option 1 — Render (recommended, no PRO, no card, no GitHub):**

A public, anonymously-clonable mirror of this repo is published at
<https://huggingface.co/datasets/senku111/pfa-source>, so Render can build
straight from that URL.

1. <https://dashboard.render.com> → sign up (email is fine; no card).
2. **New → Web Service → Public Git Repository**, URL:
   `https://huggingface.co/datasets/senku111/pfa-source`
3. Runtime **Docker**, plan **Free**, nearest region, health check
   `/api/health`.
4. Add environment variables:
   * `HF_TOKEN`
   * `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`)
   * `PFA_SHARED_REPO` = `senku111/embryo-3d`
   * `PFA_RUNS_DIR` = `/data/pfa-runs`
5. **Create Web Service**, wait for the first build (~3–5 min), open the URL.

Publish code changes to the mirror with:

```bash
export HF_TOKEN=hf_...
tools/publish_source.sh                  # single clean commit, no stale PDF
```

then click **Manual Deploy → Deploy latest commit** in Render (public-URL
services do not auto-deploy).

*Have a GitHub account?* Push the repo there instead and use the included
[`render.yaml`](render.yaml) Blueprint (**New → Blueprint**) — same service,
with auto-deploy on push.

Notes: Render injects `$PORT` (the Dockerfile honours it), free services sleep
after ~15 min idle (first visit ~30–50 s), and the disk is ephemeral — run state
is mirrored to `runs/<id>/` on the Hub and auto-resumed, so this is safe.

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
3. Press **Dry run (prepare only)** → status reaches `done` in a few seconds
   without touching Kaggle (registration is skipped when the volume already
   exists).
4. Press **Run** → the first stage appears as a kernel under
   `kernels/list?search=pfa-`. The kernel slug follows the notebook, so the
   runner polls the slug Kaggle reports rather than a guessed one.

---

## 3. Day-to-day use

| Who | What |
| --- | --- |
| Researcher (no terminal) | open the web app, pick a case, **Plan**, **Run**, follow the status card, open the viewer link |
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
  is mirrored to `runs/<run_id>/` on the Hub.
* **Large shared datasets** — the viewer fetches from the Hub by URL; nothing
  is proxied through the web app, so the web tier stays free.

---

## 5. Operational checklist

```
[ ] Web-app env vars: HF_TOKEN + KAGGLE_API_TOKEN  (or KAGGLE_USERNAME + KAGGLE_KEY)
[ ] Kaggle secret HF_TOKEN in the executing account + GPU enabled
[ ] Repos exist (or the researcher's repos are filled in on the form)
[ ] GET /api/health → ok
[ ] Dry run → done
[ ] Real run → kernels visible in Kaggle, artifacts visible on the Hub
[ ] /viewer/?specimen=<id> renders the bundle
```
