import gradio as gr
from GIS_map import OUTCOME_METRICS, plot_outcome_map, COUNTIES
from pipeline import TOP_K_DEFAULT, top_queries, analyze_file, init_engine, ask


# Build Gradio UI
def ui():
    with gr.Blocks() as demo:
        gr.Image(
            "/content/drive/MyDrive/CalWorks/Vector Database/Asset/cdss-logo.png",
            show_label=False,
            width=500)
        gr.Markdown("### CalWORKs County QA System")

        with gr.Row():
            emb_dd = gr.Dropdown(
                ["MiniLM","OpenAI Embeddings"],
                value="MiniLM",
                label="Embedding Backend")
            emb_model = gr.Textbox(
                value="sentence-transformers/all-MiniLM-L6-v2",
                label="Embed Model")

        with gr.Row():
            llm_dd = gr.Dropdown(
                ["Ollama", "OpenAI"],
                value="Ollama",
                label="LLM Backend")
            llm_model = gr.Textbox(value="smollm:135m", label="LLM Model")

        with gr.Row():
            qbox = gr.Textbox(label="Question")
            extbox = gr.Textbox(label="Web Search Query")
            extflag = gr.Checkbox(label="Include Web Search", value=False)
            topk = gr.Slider(1,20,value=TOP_K_DEFAULT,label="Top K")
            go = gr.Button("Answer")

        answer = gr.Textbox(label="Answer", lines=10)
        topq = gr.Textbox(label="Top Queries", lines=10)
        extinfo = gr.Textbox(label="External Info", lines=10)

        go.click(ask,
                inputs=[
                    qbox,
                    topk,
                    extflag,
                    extbox,
                    emb_dd,
                    emb_model,
                    llm_dd,
                    llm_model],
                outputs=[answer, topq, extinfo]
                )

        with gr.Row():
            gr.Markdown("### Upload a File")
            file_upload = gr.File(label="File", type="filepath")
            query_input = gr.Textbox(label="Optional Question")
            analyze_button = gr.Button("Analyze")
            file_out = gr.Textbox(label="Summary", lines=6)

# Optional Question is not doing anything now
        analyze_button.click(analyze_file,
                             inputs=[file_upload, query_input],
                             outputs=file_out)

        with gr.Column():
            gr.Markdown("### Top Queries")
            gr.Textbox(value=top_queries(), interactive=False, lines=10)

        with gr.Column():
            gr.Markdown("### County Outcome Map")
            metric_dd = gr.Dropdown(
                choices=OUTCOME_METRICS,
                value=OUTCOME_METRICS[0],
                label="Metric")
            county_dd = gr.Dropdown(
                choices=COUNTIES,
                multiselect=True,
                label="Counties")
            map_plot = gr.Plot(label="Map")
            demo.load(plot_outcome_map,
                      inputs=[metric_dd, county_dd],
                      outputs=map_plot)
            metric_dd.change(plot_outcome_map,
                             inputs=[metric_dd, county_dd],
                             outputs=map_plot)
            county_dd.change(plot_outcome_map,
                             inputs=[metric_dd, county_dd],
                               outputs=map_plot)

        gr.Image(
            "/content/drive/MyDrive/CalWorks/Vector Database/Asset/calworks_logo.jpeg",
            show_label=False,
            width=1600)
        placeholder = gr.Textbox(visible=False)

        demo.load(
            fn=lambda: init_engine(
                "MiniLM",
                "sentence-transformers/all-MiniLM-L6-v2",
                "Ollama",
                "smollm:135m"
            ),
            inputs=[],
            outputs=[placeholder]
        )
        demo.launch(debug=True, share=True)

    return demo

if __name__ == "__main__":
    # start_ollama()
    ui()