# Projeto de Segmentação de Imagens Médicas

Este projeto aplica várias técnicas de processamento de imagens para segmentar tecidos em imagens médicas DICOM.

## Instalação

Para instalar as dependências do projeto, é recomendado usar um ambiente virtual. Depois de clonar o repositório e criar/ativar seu ambiente virtual, você pode instalar as dependências com:

```sh
pip install -r requirements.txt
```

## Execução
Depois de instalar as dependências, você pode executar o script principal do seguinte modo:

```sh
python run.py
```

## Visualização de Resultados

Os resultados também podem ser visualizados de forma mais interativa no Jupyter Notebook "tarefa.ipynb". Para visualizar este notebook, você pode utilizar o Jupyter que já está instalado como uma das dependências e executar:

```sh
jupyter notebook tarefa.ipynb
```

O script principal irá carregar as imagens DICOM do diretório dicon, segmentar os tecidos e salvar as imagens segmentadas no diretório out.

## Contribuição
Se você gostaria de contribuir para este projeto, sinta-se à vontade para fazer um fork e enviar um pull request.
