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

_SERVICE_DOWN_MSG = "⚠️ Le service de triage est momentanément indisponible, réessayez."


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


def _request(method: str, path: str, *, json: dict | None = None, timeout: float) -> dict:
    """Call the API; on any HTTP/transport error return a sentinel {"detail": ...} dict so
    callers can render a friendly message instead of raising into the Gradio event loop."""
    try:
        r = httpx.request(method, f"{SERVICE_URL}{path}", json=json, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        return {"detail": f"service error: {exc}"}


def _post(path: str, payload: dict) -> dict:
    return _request("POST", path, json=payload, timeout=120)


def _get(path: str) -> dict:
    return _request("GET", path, timeout=30)


def _start(lang: str):
    r = _post("/session/start", {"lang": lang})
    if r.get("detail") or "question" not in r:
        return "", [(None, _SERVICE_DOWN_MSG)]
    return r["session_id"], [(None, r["question"])]


def _answer(message: str, history: list, session_id: str, lang: str):
    if not session_id:
        session_id, history = _start(lang)
    r = _post("/session/answer", {"session_id": session_id, "answer": message})
    if r.get("detail") or not (r.get("done") or r.get("question")):
        # Service error or malformed payload → friendly message, never a blank verdict card.
        return history + [(message, _SERVICE_DOWN_MSG)], "", session_id
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
        # Static greeting; the session is bootstrapped lazily on the first answer (no API call
        # on page load, so a page refresh never fails if the backend is still warming up).
        gr.Markdown("👋 _Décrivez le **motif de consultation** pour démarrer le triage._")
        chatbot = gr.Chatbot(label="Questionnaire de triage", height=380)
        msg = gr.Textbox(label="Votre réponse", placeholder="Décrivez les symptômes…")
        with gr.Row():
            send = gr.Button("Envoyer", variant="primary")
            restart = gr.Button("Nouvelle consultation")
        with gr.Accordion("Traçabilité — dossier (anonymisé)", open=False):
            trace_out = gr.JSON(label="Dossier SIH / historique")
            refresh = gr.Button("Rafraîchir le dossier")

        send.click(_answer, [msg, chatbot, session, lang], [chatbot, msg, session])
        msg.submit(_answer, [msg, chatbot, session, lang], [chatbot, msg, session])
        restart.click(_start, inputs=lang, outputs=[session, chatbot])
        refresh.click(lambda sid: _get(f"/session/{sid}"), inputs=session, outputs=trace_out)

    return demo


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", "7860")))


if __name__ == "__main__":
    main()
