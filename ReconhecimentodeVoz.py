import tkinter as tk
import speech_recognition as sr
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Função para desenhar o gráfico da forma de onda
def desenhar_grafico(onda, ax):
    ax.clear()
    ax.plot(onda)
    ax.set_title("Forma de Onda")
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Amplitude")

# Função para atualizar o gráfico
def atualizar_graficos():
    if voz_padrao:
        try:
            onda, sr = librosa.load(voz_padrao, sr=None)
            desenhar_grafico(onda, ax1)
            canvas1.draw()
        except Exception as e:
            status_label.config(text=f"Erro ao atualizar o gráfico da voz padrão: {e}")
    
    if voz_comparativa:
        try:
            onda, sr = librosa.load(voz_comparativa, sr=None)
            desenhar_grafico(onda, ax2)
            canvas2.draw()
        except Exception as e:
            status_label.config(text=f"Erro ao atualizar o gráfico da voz comparativa: {e}")

# Função para gravar a voz padrão
def gravar_voz_padrao():
    global voz_padrao
    r = sr.Recognizer()
    botao_voz_padrao.config(state=tk.DISABLED)  # Desativa o botão durante a gravação
    status_label.config(text="Gravando voz padrão... Fale agora.")
    janela.update()  # Atualiza a interface para mostrar a mudança imediatamente
    with sr.Microphone() as source:
        try:
            audio = r.listen(source)
            with open("voz_padrao.wav", "wb") as f:
                f.write(audio.get_wav_data())
            voz_padrao = "voz_padrao.wav"
            status_label.config(text="Gravação da voz padrão concluída.")
            atualizar_graficos()  # Atualiza os gráficos com a gravação da voz padrão
            if voz_comparativa is not None:
                comparar_vozes()
        except Exception as e:
            status_label.config(text=f"Erro ao gravar a voz padrão: {e}")
        finally:
            botao_voz_padrao.config(state=tk.NORMAL)  # Reativa o botão após a gravação

# Função para gravar a voz comparativa
def gravar_voz_comparativa():
    global voz_comparativa
    r = sr.Recognizer()
    botao_voz_comparativa.config(state=tk.DISABLED)  # Desativa o botão durante a gravação
    status_label.config(text="Gravando voz comparativa... Fale agora.")
    janela.update()  # Atualiza a interface para mostrar a mudança imediatamente
    with sr.Microphone() as source:
        try:
            audio = r.listen(source)
            with open("voz_comparativa.wav", "wb") as f:
                f.write(audio.get_wav_data())
            voz_comparativa = "voz_comparativa.wav"
            status_label.config(text="Gravação da voz comparativa concluída.")
            atualizar_graficos()  # Atualiza os gráficos com a gravação da voz comparativa
            if voz_padrao is not None:
                comparar_vozes()
        except Exception as e:
            status_label.config(text=f"Erro ao gravar a voz comparativa: {e}")
        finally:
            botao_voz_comparativa.config(state=tk.NORMAL)  # Reativa o botão após a gravação

# Função para comparar as vozes
def comparar_vozes():
    if voz_padrao is None or voz_comparativa is None:
        return  # Não faz nada se não tiver as duas vozes gravadas

    try:
        # Carregar as gravações
        voz_padrao_data, sr_padrao = librosa.load(voz_padrao, sr=None)
        voz_comparativa_data, sr_comparativa = librosa.load(voz_comparativa, sr=None)

        # Garantir que ambas tenham a mesma taxa de amostragem
        if sr_padrao != sr_comparativa:
            status_label.config(text="Erro: As gravações devem ter a mesma taxa de amostragem.")
            return

        # Comparar as gravações usando cross-correlation
        corr = np.correlate(voz_padrao_data, voz_comparativa_data, mode='valid')
        similaridade = np.max(corr) / (np.sqrt(np.sum(voz_padrao_data**2)) * np.sqrt(np.sum(voz_comparativa_data**2)))

        # Converter similaridade para porcentagem
        porcentagem = similaridade * 100
        resultado_label.config(text=f"Porcentagem de similaridade: {porcentagem:.2f}%")
        
    except Exception as e:
        status_label.config(text=f"Erro ao comparar as vozes: {e}")

# Função para criar a interface gráfica
def criar_interface():
    global resultado_label, voz_padrao, voz_comparativa, botao_voz_padrao, botao_voz_comparativa, status_label, canvas1, canvas2, ax1, ax2
    voz_padrao = None
    voz_comparativa = None
    
    # Criar a janela principal
    global janela
    janela = tk.Tk()
    janela.title("Reconhecimento de voz")
    
    # Definir o tamanho da janela
    window_width = 700
    window_height = 460
    
    # Obter as dimensões da tela
    screen_width = janela.winfo_screenwidth()
    screen_height = janela.winfo_screenheight()
    
    # Calcular a posição central da tela
    position_x = int((screen_width - window_width) / 2)
    position_y = int((screen_height - window_height) / 2)
    
    # Definir a geometria da janela
    janela.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
    
    # Configurar o layout da grid
    janela.grid_rowconfigure(0, weight=0)
    janela.grid_rowconfigure(1, weight=0)
    janela.grid_rowconfigure(2, weight=0)
    janela.grid_rowconfigure(3, weight=0)
    janela.grid_rowconfigure(4, weight=0)
    janela.grid_rowconfigure(5, weight=0)  # Linha para os gráficos
    janela.grid_rowconfigure(6, weight=0)  # Linha para o resultado
    janela.grid_rowconfigure(7, weight=1)  # Expansão da área de status
    janela.grid_columnconfigure(0, weight=1)
    janela.grid_columnconfigure(1, weight=1)
    
    # Título e data na mesma linha
    titulo = tk.Label(janela, text="Reconhecimento de voz", font=("Arial", 16))
    titulo.grid(row=0, column=0, sticky='w', padx=10, pady=5)  # Alinhado à esquerda
    
    subtitulo = tk.Label(janela, text="Rhaissa Rodrigues Rocha", font=("Arial", 12))
    subtitulo.grid(row=1, column=0, columnspan=2, sticky='w', padx=10)  # Abaixo do título
    
    data = tk.Label(janela, text="22/08/2024", font=("Arial", 12))
    data.grid(row=0, column=1, sticky='e', padx=10, pady=5)  # Alinhado à direita
    
    # Botões
    botao_voz_padrao = tk.Button(janela, text=" Gravar Voz Padrão", bg="#c1a0e0", fg="white", font=("Arial", 12), command=gravar_voz_padrao)
    botao_voz_padrao.grid(row=2, column=0, columnspan=2, pady=5, padx=25, sticky='ew')
    
    botao_voz_comparativa = tk.Button(janela, text=" Gravar Voz Comparativa", bg="#c1a0e0", fg="white", font=("Arial", 12), command=gravar_voz_comparativa)
    botao_voz_comparativa.grid(row=3, column=0, columnspan=2, pady=5, padx=25, sticky='ew')
    
    # Configurar os gráficos
    fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=100)  # Ajusta o tamanho do gráfico
    canvas1 = FigureCanvasTkAgg(fig1, master=janela)
    canvas_widget1 = canvas1.get_tk_widget()
    canvas_widget1.grid(row=5, column=0, padx=10, pady=10, sticky='nsew')
    
    fig2, ax2 = plt.subplots(figsize=(4, 2), dpi=100)  # Ajusta o tamanho do gráfico
    canvas2 = FigureCanvasTkAgg(fig2, master=janela)
    canvas_widget2 = canvas2.get_tk_widget()
    canvas_widget2.grid(row=5, column=1, padx=10, pady=10, sticky='nsew')
    
    # Área de exibição de resultados
    resultado_frame = tk.Frame(janela, bg="#c86dd7", width=450, height=100)
    resultado_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky='nsew')  # Espaço acima da área de resultados
    
    resultado_label = tk.Label(resultado_frame, text="Porcentagem da similaridade: 0%", bg="#c86dd7", fg="white", font=("Arial", 12))
    resultado_label.pack(pady=10)
    
    # Status da aplicação
    status_label = tk.Label(janela, text="Pronto para gravação.", font=("Arial", 10))
    status_label.grid(row=7, column=0, columnspan=2, pady=3)
    
    janela.mainloop()

# Chamar a função para criar a interface
criar_interface()
