import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import zipfile
import io
import os
import tempfile
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from collections import Counter

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Reshape,
    Lambda,
    Concatenate,
    multiply,
    add,
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Layer


class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(
            channel // self.ratio, activation="relu", kernel_initializer="he_normal"
        )
        self.shared_layer_two = Dense(channel, kernel_initializer="he_normal")
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(
            inputs
        )
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = add([avg_pool, max_pool])
        attention = Activation("sigmoid")(attention)
        return multiply([inputs, attention])


class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(
            1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
        concat = Concatenate()([avg_pool, max_pool])
        attention = self.conv(concat)
        return multiply([inputs, attention])


def cbam_module(input_tensor):
    x = ChannelAttention()(input_tensor)
    x = SpatialAttention()(x)
    return x


def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    filters = init.shape[-1]
    se = GlobalAveragePooling2D()(init)
    se = Reshape((1, 1, filters))(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)
    return multiply([init, se])


def residual_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    shortcut = x
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_regularizer=l1_l2(1e-5, 1e-4),
        name=f"{name}_conv1" if name else None,
    )(x)
    x = BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = Activation("relu", name=f"{name}_act1" if name else None)(x)
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        use_bias=use_bias,
        kernel_regularizer=l1_l2(1e-5, 1e-4),
        name=f"{name}_conv2" if name else None,
    )(x)
    x = BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    input_channels = shortcut.shape[-1]
    if stride > 1 or input_channels != filters:
        shortcut = Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=use_bias,
            kernel_regularizer=l1_l2(1e-5, 1e-4),
            name=f"{name}_shortcut_conv" if name else None,
        )(shortcut)
        shortcut = BatchNormalization(name=f"{name}_shortcut_bn" if name else None)(
            shortcut
        )
    x = add([x, shortcut], name=f"{name}_add" if name else None)
    x = Activation("relu", name=f"{name}_out" if name else None)(x)
    return x


@st.cache_resource
def load_advanced_model(h5_path):
    return load_model(
        h5_path,
        custom_objects={
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
            "cbam_module": cbam_module,
            "squeeze_excite_block": squeeze_excite_block,
            "residual_block": residual_block,
        },
    )


st.set_page_config(page_title="Phân loại sản phẩm thời trang", layout="wide")
st.title("📦 Phân loại sản phẩm thời trang và thống kê tồn kho")

model_file = st.sidebar.file_uploader("🔍 Tải mô hình CNN (.h5)", type="h5")
zip_file = st.sidebar.file_uploader("🗂 Tải file ZIP chứa ảnh sản phẩm", type="zip")
labels_text = st.sidebar.text_area(
    "🏷 Danh sách nhãn sản phẩm (mỗi dòng là 1 nhãn)",
    value="Áo thun / Áo ngắn tay\nQuần dài\nÁo len chui đầu\nVáy liền thân\nÁo khoác\nDép quai\nÁo sơ mi\nGiày thể thao\nTúi xách\nBốt cổ ngắn",
)
labels = [label.strip() for label in labels_text.split("\n") if label.strip()]


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32")
    img_array = 255 - img_array  # đảo ngược màu nếu nền trắng
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28, 1)


def predict_class(image, model, labels):
    x = preprocess_image(image)
    y = model.predict(x, verbose=0)[0]
    idx = np.argmax(y)
    return labels[idx], float(y[idx])


def get_excel_download_link(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Tải Excel</a>'


if model_file and zip_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file.read())
        model = load_advanced_model(tmp.name)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        results = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    path = os.path.join(root, file)
                    with Image.open(path) as img:
                        label, conf = predict_class(img, model, labels)
                        results.append(
                            {
                                "Tên file": file,
                                "Loại sản phẩm": label,
                                "Độ tin cậy": round(conf, 4),
                            }
                        )

        df = pd.DataFrame(results)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Kết quả phân loại")
            st.dataframe(df, use_container_width=True)

        summary = df["Loại sản phẩm"].value_counts().reset_index()
        summary.columns = ["Loại sản phẩm", "Số lượng"]

        with col2:
            st.subheader("📊 Thống kê tồn kho")
            st.dataframe(summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["Loại sản phẩm"], summary["Số lượng"], color="skyblue")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown(
            get_excel_download_link(df, "chi_tiet_phan_loai.xlsx"),
            unsafe_allow_html=True,
        )
        st.markdown(
            get_excel_download_link(summary, "thong_ke_ton_kho.xlsx"),
            unsafe_allow_html=True,
        )
else:
    st.info("Vui lòng tải mô hình .h5 và file ZIP ảnh để bắt đầu.")
