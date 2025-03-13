import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Carga del modelo YOLO
model = YOLO(r'./runs/detect/train/weights/best.pt')

# Función para redimensionar la imagen al tamaño requerido por el modelo
def redimensionar_imagen(file_path, tamaño=(640, 640)):
    imagen = Image.open(file_path)
    imagen = imagen.resize(tamaño, Image.Resampling.LANCZOS)  # Redimensionar la imagen
    return imagen

# Carga y clasificación de la imagen
def cargar_y_clasificar_imagen(file_path):
    # Redimensiona la imagen antes de procesarla
    imagen_redimensionada = redimensionar_imagen(file_path)

    # Muestra la imagen redimensionada en la interfaz
    imagen_redimensionada.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(imagen_redimensionada)
    lbl_imagen.config(image=img_tk)
    lbl_imagen.image = img_tk

    # Uso del modelo YOLO para identificar
    resultados = model(imagen_redimensionada)

    # Obtener la clase de vehículo
    if resultados and len(resultados) > 0:
        detecciones = resultados[0].boxes
        if len(detecciones) > 0:
            clase_predicha_id = detecciones.cls[0].item()  # Obtener el ID de la clase
            clase_predicha = resultados[0].names[clase_predicha_id]  # Obtener el nombre de la clase

            # Mensaje especial para la clase "police car"
            if clase_predicha == "police car":
                lbl_resultado.config(text=f"Clase predicha: {clase_predicha} 🚓", fg="#205781")  # Mensaje especial para police car
            else:
                lbl_resultado.config(text=f"Clase predicha: {clase_predicha}", fg="#205781")  # Color de "Clase predicha"
        else:
            lbl_resultado.config(text="No se detectó ningún vehículo en la imagen.", fg="#4F959D")  # Color en caso de no encontrar vehículo
    else:
        lbl_resultado.config(text="No se pudo realizar la clasificación.", fg="#4F959D")  # Color para errores

# Función para subir y clasificar una imagen desde el explorador de archivos
def subir_imagen():
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        cargar_y_clasificar_imagen(file_path)

# Función para cambiar a la página de clasificación
def ir_a_clasificacion():
    frame_inicio.pack_forget()  # Ocultar la página de inicio
    frame_clasificacion.pack(fill="both", expand=True)  # Mostrar la página de clasificación

# Función para volver a la página de inicio
def volver_a_inicio():
    frame_clasificacion.pack_forget()  # Ocultar la página de clasificación
    frame_seleccion_imagenes.pack_forget()  # Ocultar la sección de selección de imágenes
    frame_inicio.pack(fill="both", expand=True)  # Mostrar la página de inicio

# Función para mostrar la sección de imágenes predefinidas
def mostrar_seleccion_imagenes():
    frame_inicio.pack_forget()  # Ocultar la página de inicio
    frame_seleccion_imagenes.pack(fill="both", expand=True)  # Mostrar la sección de selección de imágenes

# Función para manejar la selección de una imagen predefinida
def seleccionar_imagen_predefinida(file_path):
    cargar_y_clasificar_imagen(file_path)  # Clasificar la imagen seleccionada
    frame_seleccion_imagenes.pack_forget()
    frame_clasificacion.pack(fill="both", expand=True)

# Creación de la ventana principal de la interfaz
ventana = tk.Tk()
ventana.title("Clasificador de Vehículos")
ventana.geometry("1080x700")
ventana.configure(bg="#F6F8D5")

# Configuración de fuentes
fuente_titulo = ("Montserrat", 20, "bold")  # Tamaño aumentado a 20
fuente_texto = ("Montserrat", 14)  # Tamaño aumentado a 14
fuente_boton = ("Montserrat", 14, "bold")  # Tamaño aumentado a 14

# Creación frames para las páginas
frame_inicio = tk.Frame(ventana, bg="#F6F8D5")
frame_clasificacion = tk.Frame(ventana, bg="#F6F8D5")
frame_seleccion_imagenes = tk.Frame(ventana, bg="#F6F8D5")

# Página de inicio
lbl_titulo_inicio = tk.Label(
    frame_inicio,
    text="Bienvenido al Clasificador de Vehículos",
    font=fuente_titulo,
    bg="#F6F8D5",
    fg="#205781"  # Color del texto
)
lbl_titulo_inicio.pack(pady=50)

btn_iniciar = tk.Button(
    frame_inicio,
    text="Iniciar",
    command=ir_a_clasificacion,
    font=fuente_boton,
    bg="#4F959D",  # Color del botón
    fg="white",
    activebackground="#98D2C0",  # Color al hacer clic
    activeforeground="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
btn_iniciar.pack(pady=20)

btn_seleccionar_imagenes = tk.Button(
    frame_inicio,
    text="Seleccionar de imágenes predefinidas",
    command=mostrar_seleccion_imagenes,
    font=fuente_boton,
    bg="#205781",  # Color del botón
    fg="white",
    activebackground="#4F959D",  # Color al hacer clic
    activeforeground="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
btn_seleccionar_imagenes.pack(pady=20)

# Página de clasificación
lbl_titulo_clasificacion = tk.Label(
    frame_clasificacion,
    text="Clasificador de Vehículos",
    font=fuente_titulo,
    bg="#F6F8D5",
    fg="#205781"  # Color del texto
)
lbl_titulo_clasificacion.pack(pady=20)

# Botón para subir una imagen desde el explorador de archivos
btn_subir_imagen = tk.Button(
    frame_clasificacion,
    text="Subir Imagen",
    command=subir_imagen,
    font=fuente_boton,
    bg="#4F959D",  # Color del botón
    fg="white",
    activebackground="#98D2C0",  # Color al hacer clic
    activeforeground="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
btn_subir_imagen.pack(pady=20)

# Etiqueta para mostrar la imagen
lbl_imagen = tk.Label(frame_clasificacion, bg="#F6F8D5")
lbl_imagen.pack()

# Etiqueta para mostrar el resultado de la clasificación
lbl_resultado = tk.Label(frame_clasificacion, text="Clase predicha: ", font=fuente_texto, bg="#F6F8D5", fg="#205781")
lbl_resultado.pack(pady=20)

# Botón para volver a la página de inicio
btn_volver = tk.Button(
    frame_clasificacion,
    text="Volver al Inicio",
    command=volver_a_inicio,
    font=fuente_boton,
    bg="#98D2C0",  # Color del botón
    fg="white",
    activebackground="#4F959D",  # Color al hacer clic
    activeforeground="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
btn_volver.pack(pady=20)

# Sección de selección de imágenes predefinidas
lbl_titulo_seleccion = tk.Label(
    frame_seleccion_imagenes,
    text="Selecciona una imagen",
    font=fuente_titulo,
    bg="#F6F8D5",
    fg="#205781"  # Color del texto
)
lbl_titulo_seleccion.grid(row=0, column=0, columnspan=4, pady=20)  # Usar grid aquí

# Cargar las imágenes predefinidas (asegúrate de que las rutas sean correctas)
rutas_imagenes = [
    r"./image_train/Image_000007.jpg",
    r"./image_train/moto_black.jpg",
    r"./image_train/moto_red.jpg",
    r"./image_train/moto_white.jpg",
    r"./image_train/truck_green.jpg",
    r"./image_train/truck_yellow.jpg",
    r"./image_train/van_white.jpg"
]

# Muestra las imágenes en una cuadrícula
for i, ruta in enumerate(rutas_imagenes):
    imagen = Image.open(ruta)
    imagen.thumbnail((120, 120))  # Se redimensiona la imagen para que quepa en la cuadrícula
    img_tk = ImageTk.PhotoImage(imagen)

    btn_imagen = tk.Button(
        frame_seleccion_imagenes,
        image=img_tk,
        command=lambda ruta=ruta: seleccionar_imagen_predefinida(ruta),
        borderwidth=0,
        relief="flat"
    )
    btn_imagen.image = img_tk  # Mantiene una referencia para evitar que la imagen sea eliminada por el recolector de basura
    btn_imagen.grid(row=(i // 4) + 1, column=i % 4, padx=10, pady=10)

# Botón para volver a la página de inicio desde la sección de selección de imágenes
btn_volver_seleccion = tk.Button(
    frame_seleccion_imagenes,
    text="Ir al Inicio",
    command=volver_a_inicio,
    font=fuente_boton,
    bg="#98D2C0",
    fg="white",
    activebackground="#4F959D",
    activeforeground="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
btn_volver_seleccion.grid(row=3, column=0, columnspan=4, pady=20)

# Muestra la página de inicio
frame_inicio.pack(fill="both", expand=True)

ventana.mainloop()