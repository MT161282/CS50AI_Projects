import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype(
    "assets/fonts/OpenSans-Regular.ttf",
    28
)

GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    inputs = tokenizer(
        text,
        return_tensors="tf"
    )

    mask_token_index = get_mask_token_index(
        tokenizer.mask_token_id,
        inputs
    )

    if mask_token_index is None:
        sys.exit(
            f"Input must include mask token "
            f"{tokenizer.mask_token}."
        )

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)

    result = model(
        **inputs,
        output_attentions=True
    )

    # Generate predictions
    mask_token_logits = result.logits[
        0,
        mask_token_index
    ]

    top_tokens = tf.math.top_k(
        mask_token_logits,
        K
    ).indices.numpy()

    for token in top_tokens:
        predicted_word = tokenizer.decode([token])

        print(
            text.replace(
                tokenizer.mask_token,
                predicted_word
            )
        )

    # Visualize attentions
    visualize_attentions(
        inputs.tokens(),
        result.attentions
    )


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the mask token.

    Return None if the mask token is not present.
    """
    input_ids = inputs[
        "input_ids"
    ][0].numpy().tolist()

    try:
        return input_ids.index(mask_token_id)

    except ValueError:
        return None


def get_color_for_attention_score(attention_score):
    """
    Convert an attention score between 0 and 1
    into a grayscale RGB color.
    """
    color_value = round(
        float(attention_score) * 255
    )

    return (
        color_value,
        color_value,
        color_value
    )


def visualize_attentions(tokens, attentions):
    """
    Generate one diagram for every attention head
    in every attention layer.
    """
    for layer_index, layer_attentions in enumerate(
        attentions
    ):
        number_of_heads = layer_attentions.shape[1]

        for head_index in range(number_of_heads):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                layer_attentions[0][head_index]
            )


def generate_diagram(
    layer_number,
    head_number,
    tokens,
    attention_weights
):
    """
    Generate an attention diagram for a single
    attention layer and head.
    """
    image_size = (
        GRID_SIZE * len(tokens)
        + PIXELS_PER_WORD
    )

    image = Image.new(
        "RGBA",
        (image_size, image_size),
        "black"
    )

    draw = ImageDraw.Draw(image)

    # Draw tokens
    for index, token in enumerate(tokens):

        # Draw token columns
        token_image = Image.new(
            "RGBA",
            (image_size, image_size),
            (0, 0, 0, 0)
        )

        token_draw = ImageDraw.Draw(token_image)

        token_draw.text(
            (
                image_size - PIXELS_PER_WORD,
                PIXELS_PER_WORD
                + index * GRID_SIZE
            ),
            token,
            fill="white",
            font=FONT
        )

        token_image = token_image.rotate(90)

        image.paste(
            token_image,
            mask=token_image
        )

        # Draw token rows
        _, _, width, _ = draw.textbbox(
            (0, 0),
            token,
            font=FONT
        )

        draw.text(
            (
                PIXELS_PER_WORD - width,
                PIXELS_PER_WORD
                + index * GRID_SIZE
            ),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention cells
    for row in range(len(tokens)):
        y_position = (
            PIXELS_PER_WORD
            + row * GRID_SIZE
        )

        for column in range(len(tokens)):
            x_position = (
                PIXELS_PER_WORD
                + column * GRID_SIZE
            )

            color = get_color_for_attention_score(
                attention_weights[row][column]
            )

            draw.rectangle(
                (
                    x_position,
                    y_position,
                    x_position + GRID_SIZE,
                    y_position + GRID_SIZE
                ),
                fill=color
            )

    # Save diagram
    image.save(
        f"Attention_Layer{layer_number}_"
        f"Head{head_number}.png"
    )


if __name__ == "__main__":
    main()
