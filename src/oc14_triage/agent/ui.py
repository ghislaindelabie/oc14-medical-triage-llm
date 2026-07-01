"""Gradio patient-facing demo UI for the CHSA triage agent.

Thin presentation layer: it drives the adaptive questionnaire and shows the triage verdict
by calling the FastAPI `/session/*` service over HTTP (so the demo visibly exercises the
API deliverable). All logic lives behind the API and is unit-tested; here we only format.
"""

from __future__ import annotations

import os

import httpx

SERVICE_URL = os.environ.get("AGENT_SERVICE_URL", "http://localhost:8080")

_ICON = {"urgence maximale": "🔴", "urgence modérée": "🟠", "urgence différée": "🟢"}


def render_result(result: dict, lang: str = "fr") -> str:
    """Format a completed triage result as markdown (verdict, justification, reco, req-id)."""
    urg = result.get("urgency", "")
    lines = [
        f"### {_ICON.get(urg, '⚪')} Niveau d'urgence : **{urg}**",
        f"**Justification :** {result.get('justification', '')}",
        f"**Recommandation :** {result.get('recommendation', '')}",
    ]
    if result.get("red_flags"):
        lines.append(f"**Signes d'alerte détectés :** {', '.join(result['red_flags'])}")
    lines.append(f"_Réf. dossier : `{result.get('interaction_id', '?')}` · "
                 f"texte transmis (anonymisé) : {result.get('anon_text', '')}_")
    lines.append("_Cet avis ne remplace pas une consultation médicale._"
                 if result.get("disclaimer_present")
                 else "⚠️ _(avertissement manquant dans la réponse du modèle)_")
    return "\n\n".join(lines)


def _post(path: str, payload: dict) -> dict:
    return httpx.post(f"{SERVICE_URL}{path}", json=payload, timeout=120).json()


def _get(path: str) -> dict:
    return httpx.get(f"{SERVICE_URL}{path}", timeout=30).json()


def _start(lang: str):
    r = _post("/session/start", {"lang": lang})
    return r["session_id"], [(None, r["question"])]


def _answer(message: str, history: list, session_id: str, lang: str):
    if not session_id:
        session_id, history = _start(lang)
    r = _post("/session/answer", {"session_id": session_id, "answer": message})
    bot = render_result(r, lang) if r.get("done") else r.get("question", "…")
    return history + [(message, bot)], "", session_id


def build_ui():
    """Build (do not launch) the Gradio Blocks app."""
    import gradio as gr

    with gr.Blocks(title="CHSA — Agent de triage médical (POC)") as demo:
        gr.Markdown("# 🏥 CHSA — Agent de triage médical (POC)\n"
                    "_Aide à la décision pour le personnel soignant — **ne remplace pas** un "
                    "professionnel de santé. Données anonymisées (Presidio) ; chaque interaction "
                    "est tracée par un identifiant de dossier._")
        lang = gr.Radio(["fr", "en"], value="fr", label="Langue")
        session = gr.State("")
        chatbot = gr.Chatbot(label="Questionnaire de triage", height=380)
        msg = gr.Textbox(label="Votre réponse", placeholder="Décrivez les symptômes…")
        with gr.Row():
            send = gr.Button("Envoyer", variant="primary")
            restart = gr.Button("Nouvelle consultation")
        with gr.Accordion("Traçabilité — dossier (anonymisé)", open=False):
            trace_out = gr.JSON(label="Dossier SIH / historique")
            refresh = gr.Button("Rafraîchir le dossier")

        demo.load(_start, inputs=lang, outputs=[session, chatbot])
        send.click(_answer, [msg, chatbot, session, lang], [chatbot, msg, session])
        msg.submit(_answer, [msg, chatbot, session, lang], [chatbot, msg, session])
        restart.click(_start, inputs=lang, outputs=[session, chatbot])
        refresh.click(lambda sid: _get(f"/session/{sid}"), inputs=session, outputs=trace_out)

    return demo


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", "7860")))


if __name__ == "__main__":
    main()
