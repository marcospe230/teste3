import cv2
import face_recognition

# Carregar imagens conhecidas e criar codificações faciais
imagem_conhecida1 = face_recognition.load_image_file("caminho/para/imagem1.jpg")
codificacao_conhecida1 = face_recognition.face_encodings(imagem_conhecida1)[0]

imagem_conhecida2 = face_recognition.load_image_file("caminho/para/imagem2.jpg")
codificacao_conhecida2 = face_recognition.face_encodings(imagem_conhecida2)[0]

# Banco de dados de faces conhecidas
bancodados_faces_conhecidas = [
    codificacao_conhecida1,
    codificacao_conhecida2,
]

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break

    # Converter o quadro para RGB (face_recognition usa RGB)
    rgb_frame = frame[:, :, ::-1]

    # Detectar rostos no quadro
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # Comparar a codificação facial do rosto detectado com as faces conhecidas
        matches = face_recognition.compare_faces(bancodados_faces_conhecidas, face_encoding)
        
        nome_pessoa = "Desconhecido"

        # Se houver uma correspondência, identificar a pessoa correspondente
        if True in matches:
            index_match = matches.index(True)
            nome_pessoa = f"Pessoa {index_match + 1}"  # Nomeia a pessoa correspondente no banco de dados

        # Desenhar um retângulo e o nome da pessoa no quadro
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nome_pessoa, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
