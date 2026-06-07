# Running agni-v2 on an HPC cluster

These are ready-to-edit Slurm scripts. **Before submitting anything**, open each
`.slurm` file and replace the `# EDIT:` placeholders (account, partition names,
module name, project paths, EE project, key path). They are intentionally
conservative so they fail loudly rather than silently mis-running.

## University of Toronto specifics (DRAC / Vector)

**Digital Research Alliance of Canada (DRAC, "Compute Canada")** — the default:

- Access: account at `ccdb.alliancecan.ca`, apply for a role under your PI, set up
  MFA + SSH key. Every job needs `--account=def-<your-pi>` (EDIT the slurm files).
- Environment: use `hpc/setup_env_drac.sh` on a **login node** (DRAC compute nodes
  cannot reach PyPI; it builds a `--no-download` venv from the wheelhouse).
- **Earth Engine internet rule** — build the dataset only on a cluster whose
  *compute nodes* have internet:
  - HAVE internet: **Fir, Nibi, Killarney, Vulcan** -> use `hpc/build_array.slurm`.
  - NO internet: Narval, Cedar, Graham, Trillium, Rorqual (`httpproxy` does NOT
    whitelist Google/EE) -> build off-cluster or on a login node, then `rsync` up.
- Job submissions on some clusters (e.g. Trillium) must be made from `$SCRATCH`.

**Vector Institute cluster** (only if Vector-affiliated):

- Login: `vremote.vectorinstitute.ai` (Vaughan) / newer Killarney/Bon Echo.
- Partitions are GPU-type names: `--partition=a40` / `t4v2` / `rtx6000` / `cpu`,
  plus `--qos=normal`. Set `--account` per Vector's docs. Compute nodes generally
  have no internet -> same build-elsewhere rule as the no-internet DRAC clusters.

When in doubt about internet, run `hpc/check_env.sh` in an interactive job; it
prints the answer.

## The one decision that changes everything

Earth Engine needs outbound internet. Many clusters firewall compute nodes.
Find out which case you are in (run inside an interactive job):

```bash
salloc --cpus-per-task=2 --mem=4G --time=0:30:00   # EDIT: add --partition/--account if required
bash hpc/check_env.sh
exit
```

- **Compute nodes HAVE internet** -> build with `hpc/build_array.slurm` (parallel, fast).
- **Compute nodes have NO internet** -> build on a login/data-transfer node with
  `hpc/build_login.sh` inside `tmux`, OR build off-cluster and `rsync` the parquet up.
  Everything after the build (merge, enrich, occurrence labels, training) runs fine
  on compute nodes with no internet.

## Authentication (service account = unattended jobs)

For job arrays you want no-browser auth. Create a service account in your
EE-registered Google Cloud project, download its JSON key, copy it to the cluster
(`chmod 600`), and set:

```bash
export EARTHENGINE_KEY_FILE=$HOME/.secrets/agni-ee-key.json
export EARTHENGINE_PROJECT=your-gcp-project
```

The scripts below export these for you (EDIT the paths). The `ee_session` helper
auto-detects them, so no `--ee-key` flag is needed once the env vars are set.

## Typical order of operations

```
1. hpc/check_env.sh            # interactive: internet + EE + pytest
2. build the data:
   - internet on nodes:  sbatch hpc/build_array.slurm     (then sbatch hpc/merge.slurm)
   - no internet:        bash   hpc/build_login.sh        (then sbatch hpc/merge.slurm)
3. sbatch hpc/train.slurm                 # one experiment (GPU)
4. sbatch hpc/experiments_array.slurm     # many experiments/seeds in parallel
```

## Filesystem convention

Code in `$HOME/agni-modern-aaai`; data/outputs on scratch. Point the configs'
`raw_dir` / `processed_dir` / `output_dir` at `$SCRATCH/agni/...` (EDIT in the YAML).
Scratch is large and fast but is usually auto-purged and not backed up.
