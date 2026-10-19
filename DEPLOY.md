# Deployment

How to put this project in front of researchers so they can fill a form, press
**Run**, and get a 3D model back — and how to run the same thing from a terminal
or CI.

---

## 1. Recommendation

**Hugging Face Space (web form + API) + the Kaggle API for GPU execution + the
Hugging Face Hub as storage.**

| Option | Cost | GPU | Operational burden | Verdict |
| --- | --- | --- | --- | --- |
| **HF Space (CPU) → Kaggle API** — this repo | ~0 | in the researcher's own Kaggle account (their quota) | almost none: secrets, then push | **recommended** |
| HF Space with a GPU upgrade (`sdk: docker` + `t4-small`…) | ~$0.6–3.5/h while running | built in | you pay while idle; must move 7 notebooks onto the Space | overkill for a batch pipeline |
| Kaggle notebook only (no web UI) | 0 | 30 h GPU/week free | researchers must edit YAML by hand | good fallback, not a product |
| A VM (GCP/AWS/EC2/SageMaker) with the full stack | ~$50+/month | optional | you own the OS, the GPU drivers, the nnU-Net install | wrong scale for this project |
| Colab + Drive | free-ish | optional | fragile: no long-lived service, session limits | no |

Why this split is the professional choice for a research pipeline:

* **the web tier must be cheap and boring** — it only validates a form, fills in
  parameters, pushes notebooks and polls two public APIs. A free CPU Space
  covers it.
* **the heavy work is bursty and long** — registration, hours of nnU-Net
  training and GPU inference. Kaggle gives researchers GPU hours in *their own*
  account, so the project never bills anyone for an idle GPU.
* **Hugging Face is already the transport** — every stage reads its inputs from
  and writes its outputs to the Hub, so no shared filesystem is needed between
  the form, the notebooks and the viewer.

```
 browser ──► HF Space (FastAPI, CPU)                 researcher's machine / CI
               │  form → run spec                     └─ run_pipeline.py --spec …
               │  background thread
               ▼
             Kaggle API ──► kernel per notebook (GPU when needed)
               │                  │ reads HF_TOKEN from Kaggle Secrets
               │                  ▼
               │        Hugging Face Hub
               │        volumes/<id>/ slices/<id>/ viewers/<id>/
               ▼
             GET /api/runs/<id> ──► status card ──► /viewer/?specimen=<id>
```

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

### b. Hugging Face token (Space secret)

1. <https://huggingface.co/settings/tokens> → **Fine-grained** token with
   *write* access to the repositories you will use.
2. Space → **Settings → Variables and secrets → New secret**
   * name `HF_TOKEN`, value `hf_...`

Used by the Space (`pipeline/hfio.py`) to validate runs and serve the specimen
picker.

### c. Kaggle API credentials (one Space secret)

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

The Space uses these to push, poll and download kernels. Nothing is written to
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

### e. Build the Space

```bash
cd final_pfa
git init -b main
git add Dockerfile README.md space pipeline tools specs tests *.ipynb run_pipeline.py index5.html
git commit -m "PFA pipeline as a Hugging Face Space"

huggingface-cli repo create pfa-embryo3d --type space --space-sdk docker
git remote add space https://huggingface.co/spaces/<you>/pfa-embryo3d
git push space main
```

The `README.md` front matter already declares `sdk: docker`, and the `Dockerfile`
already copies the project in. The build starts immediately; the Space is a
free CPU Space (`sdk`/`template` are the only required front-matter fields).

Secrets are **Settings → Variables and secrets** — the build never sees them.

### f. Smoke test

1. Open the Space → `GET /api/health` must report `credentials ready`.
2. Press **Plan** with the default (case 1) form → you should see
   `registration` + `reconstruct`.
3. Press **Dry run (prepare only)** → status reaches `done` in a few seconds
   without touching Kaggle (registration is skipped when the volume already
   exists).
4. Press **Run** → the first stage appears as a kernel under
   `kernels/list?search=pfa-run-`.

---

## 3. Day-to-day use

| Who | What |
| --- | --- |
| Researcher (no terminal) | open the Space, pick a case, **Plan**, **Run**, follow the status card, open the viewer link |
| Researcher (terminal/CI) | `python3 run_pipeline.py --spec specs/my-run.yaml [--engine kaggle] [--resume]` |
| Maintainer | read `GET /api/runs` on the Space, or `runs/<run_id>/` on disk |
| Anyone | browse `https://huggingface.co/datasets/<repo>/tree/main/viewers/<id>` |

Resuming: every run stores `state.json`; `--resume` (or **Run with resume**) skips
the stages already marked `done`, which is how an interrupted multi-hour
training run continues after the Kaggle quota is topped up.

Artifacts are verified after every stage (`volumes/<id>`, the groups manifest,
`viewers/<id>/manifest.json`), so a kernel that "succeeds" without pushing
anything fails the run with the log path.

---

## 4. Limits and knobs

* **Kaggle GPU quota** — free accounts get a limited GPU allowance per week;
  long training runs consume it and are the most common reason to use
  `--resume`. GPU is only requested for `train_group*` and `reconstruct`.
* **Kernel timeout** — `run.timeout_s` (default 6 h per stage, 24 h for
  training specs). Configurable per Space via the form.
* **Hugging Face rate limits** — set `HF_TOKEN`; anonymous Hub calls are
  slower and eventually throttled.
* **Runs storage** — `PFA_RUNS_DIR` (defaults to `/data/pfa-runs` when the
  Space has a persistent mount, `./runs` otherwise). Logs are also under
  `runs/<run_id>/` when the runner is used directly.
* **Large shared datasets** — the viewer fetches from the Hub by URL; nothing
  is proxied through the Space, so the Space stays on the free CPU tier.

---

## 5. Operational checklist

```
[ ] Space secrets: HF_TOKEN + KAGGLE_API_TOKEN  (or KAGGLE_USERNAME + KAGGLE_KEY)
[ ] Kaggle secret HF_TOKEN in the executing account + GPU enabled
[ ] Repos exist (or the researcher's repos are filled in on the form)
[ ] GET /api/health → ok
[ ] Dry run → done
[ ] Real run → kernels visible in Kaggle, artifacts visible on the Hub
[ ] /viewer/?specimen=<id> renders the bundle
```
