import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Inicialização segura da nova IA (Ultralytics YOLO Pose)
@st.cache_resource
def carregar_modelo_ia():
    from ultralytics import YOLO
    return YOLO('yolov8n-pose.pt')

try:
    modelo = carregar_modelo_ia()
    ia_disponivel = True
except Exception:
    ia_disponivel = False

def calcular_angulo(p1, p2, p3):
    """Calcula o ângulo entre três pontos (p2 é o vértice)."""
    a = np.array(p1)
    b = np.array(p2) # Vértice
    c = np.array(p3)
    
    radianos = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianos*180.0/np.pi)
    
    if angulo > 180.0:
        angulo = 360-angulo
    return round(angulo, 1)

# Configuração da página do Streamlit
st.set_page_config(page_title="IA Biomecânica - Overhead Squat", layout="wide")

st.title("🏋️‍♂️ Avaliação Biomecânica Automatizada por IA")
st.subheader("Detecção de Ângulos e Laudo Clínico Automático")
st.markdown("---")

# Menu lateral
st.sidebar.header("📊 Dados da Avaliação")
nome_paciente = st.sidebar.text_input("Nome do Paciente/Atleta:", "Paciente Padrão")
tipo_vista = st.sidebar.selectbox("Vista da Foto:", ["Lateral (Perfil)", "Anterior (Frente)", "Posterior (Costas)"])

uploaded_file = st.file_uploader("Suba a foto do agachamento para análise automática...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    annotated_image = img_array.copy()
    
    if ia_disponivel:
        # Rodar a Inteligência Artificial na foto
        resultados = modelo(img_array, verbose=False)
        
        if len(resultados[0].keypoints) > 0 and resultados[0].keypoints.xy is not None:
            # Extrair pontos mapeados pela IA
            kpts = resultados[0].keypoints.xy[0].cpu().numpy()
            
            # Mapeamento de índices do modelo YOLO Pose
            # 5: Ombro E, 6: Ombro D, 11: Quadril E, 12: Quadril D, 13: Joelho E, 14: Joelho D, 15: Tornozelo E, 16: Tornozelo D
            try:
                pontos = {
                    "ombro_esq": [int(kpts[5][0]), int(kpts[5][1])],
                    "ombro_dir": [int(kpts[6][0]), int(kpts[6][1])],
                    "quadril_esq": [int(kpts[11][0]), int(kpts[11][1])],
                    "quadril_dir": [int(kpts[12][0]), int(kpts[12][1])],
                    "joelho_esq": [int(kpts[13][0]), int(kpts[13][1])],
                    "joelho_dir": [int(kpts[14][0]), int(kpts[14][1])],
                    "tornozelo_esq": [int(kpts[15][0]), int(kpts[15][1])],
                    "tornozelo_dir": [int(kpts[16][0]), int(kpts[16][1])],
                }
                
                analise_clinica = ""
                cor_alerta = "green"
                
                # 1. ANÁLISE VISTA LATERAL (AUTOMÁTICA)
                if "Lateral" in tipo_vista:
                    # Usa o lado esquerdo como padrão para o perfil
                    angulo_joelho = calcular_angulo(pontos["quadril_esq"], pontos["joelho_esq"], pontos["tornozelo_esq"])
                    p_vertical = [pontos["quadril_esq"][0], pontos["quadril_esq"][1] - 100]
                    angulo_tronco = calcular_angulo(pontos["ombro_esq"], pontos["quadril_esq"], p_vertical)
                    
                    # Desenhar os eixos automáticos na foto
                    cv2.line(annotated_image, tuple(pontos["ombro_esq"]), tuple(pontos["quadril_esq"]), (255, 0, 0), 4)
                    cv2.line(annotated_image, tuple(pontos["quadril_esq"]), tuple(pontos["joelho_esq"]), (0, 255, 0), 4)
                    cv2.line(annotated_image, tuple(pontos["joelho_esq"]), tuple(pontos["tornozelo_esq"]), (0, 0, 255), 4)
                    
                    for pt in [pontos["ombro_esq"], pontos["quadril_esq"], pontos["joelho_esq"], pontos["tornozelo_esq"]]:
                        cv2.circle(annotated_image, tuple(pt), 8, (255, 255, 255), -1)
                        
                    cv2.putText(annotated_image, f"{angulo_joelho} deg", (pontos["joelho_esq"][0] + 15, pontos["joelho_esq"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"{angulo_tronco} deg", (pontos["quadril_esq"][0] - 80, pontos["quadril_esq"][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    if angulo_tronco > 45:
                        analise_clinica = f"⚠️ **Inclinação Excessiva do Tronco ({angulo_tronco}°):** Sugere restrição de mobilidade em flexão dorsal de tornozelo ou fadiga/fraqueza dos eretores da espinha e glúteo máximo."
                        cor_alerta = "orange"
                    else:
                        analise_clinica = f"✅ **Alinhamento de Tronco Adequado ({angulo_tronco}°):** Boa estabilidade lombo-pélvica e controle excêntrico durante o agachamento."
                
                # 2. ANÁLISE VISTA ANTERIOR / POSTERIOR (AUTOMÁTICA)
                else:
                    p_horiz_quadril = [pontos["quadril_esq"][0] + 100, pontos["quadril_esq"][1]]
                    Inclinacao_pelvica = calcular_angulo(pontos["quadril_dir"], pontos["quadril_esq"], p_horiz_quadril)
                    if Inclinacao_pelvica > 90: Inclinacao_pelvica = np.abs(180 - Inclinacao_pelvica)
                    
                    cv2.line(annotated_image, tuple(pontos["ombro_dir"]), tuple(pontos["ombro_esq"]), (255, 165, 0), 4)
                    cv2.line(annotated_image, tuple(pontos["quadril_dir"]), tuple(pontos["quadril_esq"]), (128, 0, 128), 4)
                    
                    for pt in [pontos["ombro_dir"], pontos["ombro_esq"], pontos["quadril_dir"], pontos["quadril_esq"]]:
                        cv2.circle(annotated_image, tuple(pt), 8, (255, 255, 255), -1)
                        
                    cv2.putText(annotated_image, f"Desvio Pelvico: {Inclinacao_pelvica} deg", (pontos["quadril_esq"][0] - 50, pontos["quadril_esq"][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if Inclinacao_pelvica > 3.0:
                        analise_clinica = f"⚠️ **Shift Lateral / Assimetria Pélvica ({Inclinacao_pelvica}°):** Indica dominância de descarga de peso em um dos membros, sugerindo assimetria de força ou encurtamento assimétrico de adutores/banda iliotibial."
                        cor_alerta = "orange"
                    else:
                        analise_clinica = f"✅ **Simetria Pélvica Excelente ({Inclinacao_pelvica}°):** Distribuição de carga perfeitamente uniforme entre as estruturas cinéticas."

                # Apresentação na Tela
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Foto Original", use_container_width=True)
                with col2:
                    st.image(annotated_image, caption="Análise Automática por IA", use_container_width=True)
                    
                    # Botão para baixar
                    result_img = Image.fromarray(annotated_image)
                    buf = io.BytesIO()
                    result_img.save(buf, format="JPEG")
                    st.download_button(label="💾 Baixar Imagem Analisada", data=buf.getvalue(), file_name=f"resultado_{nome_paciente}.jpg", mime="image/jpeg")
                
                # Laudo
                st.markdown("---")
                st.subheader(f"📋 Diagnóstico Clínico Automatizado: {nome_paciente}")
                if cor_alerta == "green":
                    st.success(analise_clinica)
                else:
                    st.warning(analise_clinica)
                    
                st.markdown("""
                **Condutas Clínicas Recomendadas para Correção:**
                * Ativação e ganho de força excêntrica de Glúteo Médio e Glúteo Máximo.
                * Trabalho de mobilidade de tornozelo (liberação miofascial de tríceps sural).
                * Treinamento de controle motor com feedback visual.
                """)
                
            except Exception:
                st.error("❌ A IA não conseguiu mapear todos os pontos anatômicos necessários nesta foto. Garanta boa iluminação e roupas justas.")
        else:
            st.error("❌ Nenhum corpo humano foi detectado na imagem. Tente outra foto.")
    else:
        st.error("O motor de IA está carregando ou falhou na inicialização.")