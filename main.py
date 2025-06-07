import cv2

# === Cargar imagen ===
img = cv2.imread("input/imagen.jpg")
if img is None:
    raise ValueError("No se encontró la imagen en la carpeta 'input'.")

# Redimensionar para vista previa rápida (opcional)
img = cv2.resize(img, (800, 600))

# === Aplicar filtro estilo cartoon ===

# 1. Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Aplicar blur para suavizar
gray_blur = cv2.medianBlur(gray, 7)

# 3. Detección de bordes con adaptative threshold
edges = cv2.adaptiveThreshold(
    gray_blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=9,
    C=2
)

# 4. Aplicar filtro bilateral para suavizar colores
color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)

# 5. Combinar bordes + color
cartoon = cv2.bitwise_and(color, color, mask=edges)

# === Guardar resultado ===
cv2.imwrite("output/imagen_cartoon.jpg", cartoon)
print("✅ Imagen procesada y guardada en 'output/imagen_cartoon.jpg'")
