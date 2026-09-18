"""Portable, offline HTML and a reproducible ZIP artifact."""

import zipfile
from datetime import datetime, timezone
from io import BytesIO

from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs

from . import __version__
from .charts import comparison_chart, report_charts
from .profile import compare_profiles, profile_data, resolve_roles
from .serialization import dumps


def build_report(original, result, name="Dataset", metadata=None):
    roles = resolve_roles(original, result.config.get("roles"))
    after_roles = {k: v for k, v in roles.items() if k in result.frame}
    # Explicit cast steps update the role unless the original role is ID/ignore.
    for step in result.config.get("steps", []):
        if step["op"] == "cast":
            for col in step["columns"]:
                if col in after_roles and after_roles[col] not in ("id", "ignore"):
                    after_roles[col] = {"numeric": "numeric", "datetime": "datetime", "string": "text"}.get(
                        step.get("dtype", "numeric"), "text"
                    )
    before = profile_data(original, roles)
    after = profile_data(result.frame, after_roles)
    sections = []
    for label, frame, mapping in [("Dữ liệu gốc", original, roles), ("Sau xử lý", result.frame, after_roles)]:
        figures = report_charts(frame, mapping)
        sections.append(
            {
                "title": label,
                "figures": [
                    fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
                    for fig in figures
                ],
            }
        )
    comparisons = []
    for col in original:
        if col in result.frame and roles[col] == "numeric" and after_roles[col] == "numeric":
            fig = comparison_chart(original, result.frame, col)
            if fig is not None:
                comparisons.append(fig.to_html(full_html=False, include_plotlyjs=False))
            if len(comparisons) >= 6:
                break
    env = Environment(loader=PackageLoader("dataprep", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html").render(
        name=name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
        metadata=dumps(metadata or {}),
        before=before,
        after=after,
        comparison=compare_profiles(before, after).to_dict("records"),
        sections=sections,
        comparisons=comparisons,
        log=result.log,
        pipeline=dumps(result.config),
        plotly_js=get_plotlyjs(),
        stats_json=dumps({"before": before, "after": after}),
    )
    return html


def export_bundle(original, result, name="Dataset", metadata=None):
    report = build_report(original, result, name, metadata)
    schema = {c: str(result.frame[c].dtype) for c in result.frame}
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", report)
        archive.writestr("cleaned_data.csv", result.frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("pipeline.json", dumps(result.config))
        archive.writestr("processing_log.json", dumps(result.log))
        archive.writestr("schema.json", dumps(schema))
        archive.writestr("source_metadata.json", dumps(metadata or {}))
    return report, buffer.getvalue()
