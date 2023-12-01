# Shape-Aware Synthon Search (SASS)

This repo contains the code for running synthon-based ROCS queries, as described in [Shape-Aware Synthon Search (SASS) for virtual screening of synthon-based chemical spaces](https://chemrxiv.org/engage/chemrxiv/article-details/655bccc66e0ec7777f59747e).

<center>
    <img src="SASS.jpg?raw=true" width="500">
</center>

SASS is a synthon-based virtual screening method that carries out shape similarity searches in the synthon space instead of the enumerated product space. Queries are fragmented, and reaction synthons are scored against query fragments to prioritize top synthon combinations. A tiny fraction of the full library is then instantiated and scored, thereby avoiding full enumeration/scoring and significantly accelerating large-scale, shape-based virtual screening.

## Usage

Configure `config_template.yml`, and run:

```
python query_main.py
```

Note that users should install necessary third-party libraries.