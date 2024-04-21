import gradio as gr  # type: ignore
from app.tools import (  # type: ignore
    parse_arguments,
    setup_gpu,
    setup_mmseg_inference,
    threshold_heatmap_from_input_img,
)


def setup_app_thresholding(list_of_input_paths):  # , prediction_model):
    input_image = gr.Image(label="Input image")
    out_mask = gr.Image(label="Segmentation mask", show_label=True)
    out_heatmap = gr.Image(label="Uncertainty heatmap", show_label=True)
    output_images = [out_mask, out_heatmap]

    examples = [[ex] for ex in list_of_input_paths]

    title = (
        "Generate an uncertainty heatmap for a threshold λ and prediction set method."
    )

    description = """
        - Author: Luca Mossina, IRT Saint Exupéry. DEEL project, www.deel.ai
        
        Get a heatmap for a specified threshold λ ∈ [0,1] and prediction set method:

        * LAC (Least Ambiguous Set-valued Classif): take classes whose softmax is above (1-λ).
        * APS (Adaptive Prediction Set): take classes whose cumulated softmax is above λ.

        To obtain a statistical guarantee, we can pick the value of λ with **conformal prediction**,
        with a process known as *conformalization*.
        *Without conformalization*, the heatmap is a heuristic measure of uncertainty.

        For each pixel, we "activate" 1 or more labels (up to K). 

        If many classes are activated, the conformalized model gives a signal of higher uncertainty.
    """

    usecase = gr.Radio(
        choices=["Cityscapes", "ADE20K", "LoveDA"],
        label="Which dataset the predictor was trained on.",
        value="Cityscapes",
    )

    normalize_tot_num_classes = gr.Checkbox(
        label="Normalize heatmap by total number of classes (useful if few classes in ground truth)",
        value=False,
    )

    conformal_methods = gr.Radio(
        choices=["LAC", "APS"],
        label="Choose the conformal method to build prediction set and varisco map",
        value="LAC",
    )

    threshold = gr.Slider(
        minimum=0.95,
        maximum=1.0,
        step=0.0001,
        label="Choose a threshold",
        value=0.99,
    )

    app = gr.Interface(
        title=title,
        description=description,
        fn=threshold_heatmap_from_input_img,
        inputs=[
            input_image,
            conformal_methods,
            threshold,
            usecase,
            normalize_tot_num_classes,
        ],
        outputs=output_images,
        # outputs=[segmask_component, heatmap_component],
        examples=examples,
        analytics_enabled=True,
        allow_flagging="never",
        live=True,
        # cache_examples=True,
    )

    return app


if __name__ == "__main__":
    args = parse_arguments()
    device = setup_gpu(args.gpu)
    # _, _, input_paths = setup_mmseg_inference(device, args.usecase)

    RANDOM_SEED = 42

    _, _, input_paths_1 = setup_mmseg_inference(
        device, "Cityscapes", random_seed=RANDOM_SEED
    )
    _, _, input_paths_2 = setup_mmseg_inference(
        device, "ADE20K", random_seed=RANDOM_SEED
    )
    _, _, input_paths_3 = setup_mmseg_inference(
        device, "LoveDA", random_seed=RANDOM_SEED
    )

    from cose.utils import load_project_paths

    cose_path = load_project_paths().COSE_PATH
    input_paths = []
    input_paths.extend([pt for pt in input_paths_1[:5]])
    input_paths.extend(input_paths_2[:1])
    input_paths.extend(input_paths_3[:8])

    print(f" ------ Example inputs:")
    for el in input_paths:
        print(el)

    app = setup_app_thresholding(list_of_input_paths=input_paths)

    try:
        app.launch(
            share=args.share_url,
            debug=True,
            # auth=("user", "confiance"),
            show_error=True,  # frontend errors to console
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        app.close()
    except Exception as e:
        print(f"Exception: {e}")
        app.close()
