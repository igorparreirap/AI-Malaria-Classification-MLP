# Dataset de Células de Malária

Este projeto utiliza o dataset **Malaria Cell Images Dataset**, que contém imagens de células sanguíneas segmentadas para a detecção de malária.

## 📥 Download dos Dados

O dataset necessário para a execução deste projeto pode ser baixado diretamente através do link abaixo (SharePoint PUC Goiás):

👉 [**Baixar cell_images.zip**](https://pucdegoias-my.sharepoint.com/:u:/g/personal/20221003300956_pucgo_edu_br/IQDsX2wfy94VTZy_mOz2hKOOASWLuDYlRrtUP2k2LSvema8?e=gEeYW2)

## 📂 Instruções de Instalação

1. Faça o download do arquivo `.zip` no link acima.
2. Extraia o conteúdo na **raiz** deste projeto.
3. Certifique-se de que a pasta extraída se chame `cell_images` e contenha as subpastas `Parasitized` e `Uninfected`.

A estrutura final de pastas do seu projeto deve ficar exatamente assim para o código funcionar:

```text
AI-Malaria-Classification-MLP/
│
├── cell_images/           <-- A pasta extraída deve ficar AQUI
│   ├── Parasitized/
│   │   ├── C33P1thinF_IMG_20150619_114756a_cell_179.png
│   │   └── ...
│   └── Uninfected/
│       ├── C1_thinF_IMG_20150604_104722_cell_9.png
│       └── ...
│
├── data/
│   └── README.md
├── reports/
├── src/
├── .gitignore
├── README.md
└── requirements.txt