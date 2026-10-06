"""
Tests unitaires pour le parseur HTML d'articles arXiv.
Vérifie la robustesse du nettoyage, la conversion en blocs typés,
le respect des seuils d'équations, la transformation des tables en Markdown
et l'élimination stricte de la bibliographie.
"""

import pytest
from src.parser import parse_paper_html


@pytest.fixture
def sample_arxiv_html() -> str:
    return r"""
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title></head>
    <body>
        <div class="ltx_abstract">
            <h2>Abstract</h2>
            <p>This is the abstract discussing RAG and retrieval.</p>
        </div>

        <section class="ltx_section" id="S1">
            <h2 class="ltx_title">1 Introduction</h2>
            <p>Introduction text with inline math <math alttext="x \in \mathbb{R}"><mrow><mi>x</mi></mrow></math> here.</p>
            
            <figure class="ltx_figure" id="F1">
                <img src="fig1.png" alt="figure image"/>
                <figcaption class="ltx_caption">Figure 1: Overview of the RAG pipeline.</figcaption>
            </figure>

            <table class="ltx_equation" id="E1">
                <math alttext="E = mc^2"><mi>E</mi></math>
            </table>

            <table class="ltx_equation" id="E2">
                <math alttext="L = \sum_{i=1}^{N} \int_{0}^{1} \frac{\alpha \cdot \beta + \gamma \cdot \delta}{\epsilon + \zeta} d\theta + \text{a very long formula that exceeds two hundred characters to test the threshold replacement requirement specified by the user in the prompt rules strictly for portfolio compliance and performance}"></math>
            </table>
        </section>

        <section class="ltx_section" id="S2">
            <h2 class="ltx_title">2 Experimental Results</h2>
            <figure class="ltx_table" id="T1">
                <figcaption class="ltx_caption">Table 1: Main benchmark results.</figcaption>
                <table class="ltx_tabular">
                    <tr><th>Model</th><th>Recall@5</th><th>MRR</th></tr>
                    <tr><td>BM25</td><td>0.65</td><td>0.52</td></tr>
                    <tr><td>Dense</td><td>0.78</td><td>0.61</td></tr>
                </table>
            </figure>
        </section>

        <section class="ltx_bibliography" id="bib">
            <h2 class="ltx_title">References</h2>
            <p>Reference 1: Lewis et al. 2020.</p>
        </section>
    </body>
    </html>
    """


def test_parser_structure_and_types(sample_arxiv_html):
    metadata = {
        "arxiv_id": "2401.99999v1",
        "clean_id": "2401.99999",
        "title": "Synthetic Test Paper",
        "authors": ["Test Author"],
        "published": "2024-05-01",
        "categories": ["cs.CL"],
        "html_url": "https://arxiv.org/html/2401.99999v1",
    }

    result = parse_paper_html(sample_arxiv_html, metadata)

    assert result["arxiv_id"] == "2401.99999v1"
    assert result["title"] == "Synthetic Test Paper"

    # 1. Vérifier qu'aucune bibliographie n'est présente
    section_ids = [sec["section_id"] for sec in result["sections"]]
    assert "bib" not in section_ids
    for sec in result["sections"]:
        assert "References" not in sec["heading"]

    # 2. Vérifier l'Abstract
    abstract_sec = next(s for s in result["sections"] if s["section_id"] == "abstract")
    assert abstract_sec["heading"] == "Abstract"
    assert len(abstract_sec["blocks"]) == 1
    assert abstract_sec["blocks"][0]["type"] == "text"

    # 3. Vérifier S1 (Introduction, math inline, figure, equations)
    s1 = next(s for s in result["sections"] if s["section_id"] == "S1")
    block_types = [b["type"] for b in s1["blocks"]]
    assert "text" in block_types
    assert "figure" in block_types
    assert "equation" in block_types

    # Vérification inline math
    text_block = next(b for b in s1["blocks"] if b["type"] == "text")
    assert "$x \\in \\mathbb{R}$" in text_block["content"]

    # Vérification figure : légende présente, pas de balise img
    fig_block = next(b for b in s1["blocks"] if b["type"] == "figure")
    assert "[Figure 1: Overview of the RAG pipeline.]" == fig_block["content"]

    # Vérification équation courte (< 200 chars) -> $$E = mc^2$$
    eq_blocks = [b for b in s1["blocks"] if b["type"] == "equation"]
    assert len(eq_blocks) == 2
    assert eq_blocks[0]["content"] == "$$E = mc^2$$"

    # Vérification équation longue (> 200 chars) -> [EQUATION]
    assert eq_blocks[1]["content"] == "[EQUATION]"

    # 4. Vérifier S2 (Table)
    s2 = next(s for s in result["sections"] if s["section_id"] == "S2")
    table_block = next(b for b in s2["blocks"] if b["type"] == "table")
    assert "[Table 1: Main benchmark results.]" in table_block["content"]
    assert "| Model | Recall@5 | MRR |" in table_block["content"]
    assert "| BM25 | 0.65 | 0.52 |" in table_block["content"]

    # 5. Vérifier les métadonnées globales
    assert result["block_counts"]["figure"] == 1
    assert result["block_counts"]["table"] == 1
    assert result["block_counts"]["equation"] == 2
    assert result["total_words"] > 0
