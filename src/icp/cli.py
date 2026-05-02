import json
import pandas as pd
import click
from pathlib import Path
from .index import Index
from .search import find


@click.group()
def cli():
    pass


@cli.command("build-index")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", default="index", show_default=True)
def build_index(input_path: str, out_dir: str):
    df = pd.read_csv(input_path)
    if "technologies" in df.columns:
        df["technologies"] = df["technologies"].fillna("").apply(
            lambda s: [t.strip() for t in str(s).split(";") if t.strip()]
        )
    idx = Index.build(df)
    idx.save(out_dir)
    click.echo(f"built index of {len(df)} rows at {out_dir}")


@cli.command()
@click.option("--seeds", required=True, help="comma-separated domains")
@click.option("--top", default=25, show_default=True)
@click.option("--index", "idx_dir", default="index", show_default=True)
@click.option("--rerank", is_flag=True, help="LLM re-rank (requires ANTHROPIC_API_KEY)")
def cli_find(seeds: str, top: int, idx_dir: str, rerank: bool):
    idx = Index.load(idx_dir)
    seed_list = [s.strip().lower() for s in seeds.split(",") if s.strip()]
    out = find(idx, seed_list, top=top, llm_rerank=rerank)
    click.echo(json.dumps(out, indent=2))

cli.add_command(cli_find, name="find")


if __name__ == "__main__":
    cli()
