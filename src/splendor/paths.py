from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] # src/splendor/paths.py -> repo root

def ensure_parent(path: str | Path) -> Path:
    """Create the parent folder of 'path' if needed, and return it as a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Reference
REFS_DIR = Path.home() / "dev/refs/"
BIKENWLOOPS_DIR = REFS_DIR / "bikenwloops"

def bikenwloops_network_path(area: str, suffix: str = "") -> Path:
    return BIKENWLOOPS_DIR / "data/processed" / area / f"network_preprocessed{suffix}.xz"

def bikenwloops_loopcensus_path(area: str, max_nodes: int, max_length_m: int) -> Path:
    return BIKENWLOOPS_DIR / "data/processed" / area / f"loopcensus_{max_nodes}_maxlength{max_length_m}.pkl"

BIKENODEPLANNER_DIR = REFS_DIR / "bike-node-planner-data-denmark"

def bikenodeplanner_landuse_path(code: str) -> Path:
    return BIKENODEPLANNER_DIR / f"data/polygon/{code}/land_anvendelse.gpkg"


# Data
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXP = DATA / "exp"

def network_path(area: str) -> Path:
    return INTERIM / area / "network.gpkg"

def _level_dir(area: str, level: str) -> Path:
    return PROCESSED / area / level

def _level_prefix(level: str) -> str:
    return level.removesuffix("s")


# Loop census
def loop_census_path(area: str) -> Path:
    return _level_dir(area, "loops") / "loop_census.npz"

def loop_attributes_path(area: str) -> Path:
    return _level_dir(area, "loops") / "loop_attributes.csv"


# Basemap05 raw
BASEMAP05_DIR = RAW / "basemap_05_public_geotif"
BASEMAP05_LU01_RAW = BASEMAP05_DIR / "lu_01_2024.tif"
BASEMAP05_LU01_VAT = BASEMAP05_DIR / "lu_01_2024.tif.vat.dbf"

# Basemap05 interim (per study area)
def basemap05_lu01_path(area: str) -> Path:
    return INTERIM / area / "basemap05_lu01.tif"

def basemap05_main_types_path(area: str) -> Path:
    return INTERIM / area / "basemap05_main_types.tif"


# Landuse
def landuse_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_landuse_r{r}_rs{r_s}.csv"

def landuse_polygon_gaps_path(area: str) -> Path:
    return EXP / area / "landuse_polygon_gaps.gpkg"


# Components
def components_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_components_r{r}_rs{r_s}.csv"


# Score
def score_grid_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_score_grid_r{r}_rs{r_s}.csv"

def rank_summary_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_rank_summary_r{r}_rs{r_s}.csv"

def rank_correlations_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_rank_correlations_r{r}_rs{r_s}.csv"


# Splendor
def splendor_path(area: str, level: str, r: float, r_s: float) -> Path:
    return _level_dir(area, level) / f"{_level_prefix(level)}_splendor_r{r}_rs{r_s}.csv"


# Plots
def landuse_plot_path(area: str, level: str, r: float, detail: str = "") -> Path:
    suffix = f"_landuse_{detail}_r{r}.png" if detail else f"_landuse_r{r}.png"
    return PLOTS / area / (_level_prefix(level) + suffix)
