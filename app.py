import dash
from dash import Dash, dcc, html, Input, Output
from dash import dcc
from dash import html
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
from predict_func_single import preprocess_images
from food_explanation import chain
import os

IMAGE_SIZE = 300 
BATCH_SIZE = 32
preprocessing_func = eff_preprocess_input

unique_labels = np.load("label_mapping.npy", allow_pickle=True)
model = load_model('effi_batch_fix_best.keras')


stylesheets = [
    "https://cdn.jsdelivr.net/npm/reset-css@5.0.1/reset.min.css", 
    "https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap",
    "https://fonts.googleapis.com/css2?family=Gowun+Dodum&family=Montserrat:ital,wght@0,100..900;1,100..900&family=Playwrite+GB+S:ital,wght@0,100..400;1,100..400&family=Roboto+Mono:ital,wght@0,100..700;1,100..700&family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap",
    ]

app = Dash(__name__, external_stylesheets=stylesheets)
server = app.server

app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "minHeight":"100vh",
        "width": "100%",
        "fontFamily": "Montserrat, sans-serif",
    },

    children=[
        html.Header(
            style={
                "textAlign": "center", 
                "marginTop": 10,
                "marginBottom": 10, 
                "border": "2px solid black",
                "display": "flex",
                "alignItems": "center", 
                "justifyContent": "center",
                "height": "100px",
                "width": "1010px",
                "borderRadius": "10px",
                },
            children=[html.H1('Korean Cuisine', style={"fontSize": "30px", "fontFamily": "Montserrat, sans-serif"})]
        ),
        # 첫 번째 업로드 및 결과
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "flex-start",
                "width": "100%",
                "height": "350px",
                "marginBottom": "5px"
            },
            children=[
                # 왼쪽 업로드
                html.Div(
                    style={
                        "width": "500px",
                        "height": "500px",
                        "textAlign": "center",
                        "border": "1px solid #ccc",
                        "margin": "0 5px 0 0",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        "overflowX": "auto",
                    },
                    children=[
                        dcc.Upload(
                            id='upload-image',
                            children=html.Button('Upload a Image'),
                            multiple=True,
                            style={
                                'textAlign': 'center',
                                'paddingTop': '15px',
                                'cursor': 'pointer',
                                'margin': '0 auto',
                                'width': '200px',
                            },
                        ),
                        html.Div(
                            id='uploaded-image',
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "overflowX": "auto",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "flexWrap": "wrap",
                                "marginTop": "10px",
                                "gap": "10px",
                                "width": "100%",
                                })
                    ]
                ),
                # 오른쪽 예측 결과 출력
                html.Div(
                    style={
                        "width": "500px",
                        "height": "500px",
                        "textAlign": "center",
                        "border": "1px solid #ccc",
                        "margin": "0 0 0 5px",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    },
                    children=[
                        html.Div(id='output-prediction', 
                                 style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "overflowX": "auto",
                                    "justifyContent": "center",
                                    "alignItems": "center",
                                    "flexWrap": "wrap",
                                    "marginTop": "10px",
                                    "gap": "10px",
                                    "width": "100%",}),

                        html.Div(id='output-prediction-2', 
                                 style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "overflowX": "auto",
                                    "justifyContent": "center",
                                    "alignItems": "center",
                                    "flexWrap": "wrap",
                                    "marginTop": "10px",
                                    "gap": "10px",
                                    "width": "100%",})  
                    ]
                ),
            ]
        ),
    ],
)

@app.callback(
    [Output('output-prediction', 'children'),
     Output('output-prediction-2', 'children'),
     Output('uploaded-image', 'children')],
    [Input('upload-image', 'contents')],
)

def update_output(contents):
    if contents is None:
        return '', '', ''

    try:
        # 여러 이미지 처리
        if isinstance(contents, list):
            predictions = []
            images = []
            gpt_responses = []

            for content in contents:
                processed_image = preprocess_images(content)
                prediction = model.predict(processed_image)
                predicted_label_idx = np.argmax(prediction, axis=1)[0]
                predicted_label = unique_labels[predicted_label_idx]
                predictions.append(predicted_label)
                images.append(content)

                # GPT 응답 생성
                gpt_input = {"food": predicted_label}
                gpt_response = chain.invoke(gpt_input)
                gpt_responses.append(gpt_response.content)

            image_elements = [
                html.Div(
                    children=[
                        html.Img(src=image, style={"width": "300px", "height": "auto", "borderRadius": "10px"}),
                        html.P(pred_label, style={
                            "textAlign": "center",
                            "fontSize": "20px",
                            "fontWeight": "bold",
                            "marginTop": "10px",
                            "fontFamily": "Montserrat, sans-serif"}),],
                    style={"textAlign": "center"},
                )
                for image, pred_label in zip(images, predictions)
            ]

            prediction_texts = [html.P(f"Prediction: {pred_label}", style={
                "textAlign": "center", 
                "fontSize": "16px", 
                "fontWeight": "bold", 
                "marginTop": "10px", 
                "fontFamily": "Montserrat, sans-serif"
                }) for pred_label in predictions]
            
            # GPT 응답 출력
            gpt_texts = [
                    html.Div(
                        children=[
                            # Pronounced 출력
                            html.P(
                                f"{response_parts[0].strip()}" if len(response_parts) > 0 else "Pronounced: N/A",
                                style={
                                    "textAlign": "left",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                    "marginBottom": "10px",
                                    "fontFamily": "Montserrat, sans-serif"
                                }
                            ),

                            # Explain 출력 (굵게 강조)
                            html.P(
                                [
                                    html.Span("Explain: ", style={"fontWeight": "bold"}),  # 'Explain' 굵게
                                    response_parts[2].strip() if len(response_parts) > 2 else "No explanation available."
                                ],
                                style={
                                    "textAlign": "left",
                                    "fontSize": "16px",
                                    "lineHeight": "1.6",
                                    "marginBottom": "10px",
                                    "fontFamily": "Montserrat, sans-serif"
                                }
                            ),

                            # Allergy 출력 (굵게 강조)
                            html.P(
                                [
                                    html.Span("Allergy: ", style={"fontWeight": "bold"}),  # 'Allergy' 굵게
                                    response_parts[-1].strip() if len(response_parts) > 3 else "No allergy information available."
                                ],
                                style={
                                    "textAlign": "left",
                                    "fontSize": "16px",
                                    "lineHeight": "1.6",
                                    "marginBottom": "10px",
                                    "fontFamily": "Montserrat, sans-serif"
                                }
                            )
                        ],
                        style={
                            "textAlign": "left",
                            "border": "1px solid #ccc",
                            "borderRadius": "10px",
                            "padding": "10px",
                            "marginTop": "10px",
                            "backgroundColor": "#fff",
                            "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                        }
                    )
                    for response in gpt_responses
                    for response_parts in [response.split('|')]
                ]
            
            return prediction_texts, gpt_texts, image_elements

    except Exception as e:
        return f"Error: {str(e)}", '', ''



if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))




    # gpt_texts = [html.P(response, style={
    #             "textAlign": "center", 
    #             "fontSize": "16px", 
    #             "fontWeight": "bold", 
    #             "marginTop": "10px", 
    #             "fontFamily": "Montserrat, sans-serif"
    #             }) for response in gpt_responses]