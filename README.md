# Repositório do material de "live coding" da disciplina de ETE
Componente teoricó-prática da disciplina de [Econometria Temporal e Espacial](https://www.ua.pt/pt/uc/15124), do plano curricular do [Mestrado em Ciência dos Dados para Ciências Sociais](http://cdcs.web.ua.pt/?page_id=616) da [Universidade de Aveiro](https://www.ua.pt/pt/c/473/p).
Este repositório alberga os contéudos das sessões de código "ao vivo" da componente de econometria espacial. Pretende-se que este material permita consolidar os conceitos abordados  na aula, procurando  familiarizar os estudantes - na sua maioria das ciências sociais - com a utilização de métodos e técnicas de econometria espacial no contexto interdisciplinar do cientista de dados. 
A componente espacial desta disciplina assume que os alunos estão familiarizados com  a) análise estatística (inferência e modelos estatísticos de regressão) b) sistemas de informação geográfica, c) modelos de aprendizagem computacional.

Entre outros dados, este repositório recorre aos dados produzidos no âmbito do projeto [Avaliação de impacto territorial para operações de revitalização urbana](https://www.ua.pt/pt/projetos-id/1062) e da tese de doutoramento [A estrutura de interação de um sistema e-territorial: território, mercado de habitação e econometria espacial](https://ria.ua.pt/handle/10773/26275). Os dados disponibilizados encontram-se protegidos por direitos de autor  e a sua utilização fora do âmbito da disciplina das atividades desenvolvidas na disciplina de Econometria Temporal e Espacial da Universidade de Aveiro é **estritamente proibida** .

#### -- Estado do projeto: [ATIVO]




### Biblioteca indispensável
* [Pysal - Python Spatial Analysis Library](https://pysal.org/)
* Suggested reference BOOK:  [Geographic Data Science with Python](https://geographicdata.science/book/intro.html)


## Requisitos

Recomenda-se a utilização do gestor de pacotes [Miniforge](https://github.com/conda-forge/miniforge) para a instalação e configuração do ambiente de programação (Python e todas as bibliotecas necessárias).

## Instruções de utilização do repositório como template
A versão inicial deste repositório pode ser adotada como template base para a realização dos  trabalhos da disciplina. 

1. Clonar o [repositório](https://github.com/paulorlb/projETE).
2. Utilizar o ficheiro `env_ETE.yml` para clonar o ambiente de programação pré-configurado.

NOTA: Na utilização deste repositório como template para os trabalhos práticos sugere-se que cada aluno defina o nome dos seus repositórios seguindo o formato `projETE_A#####_##`. Além do elemento comum 'projETE' devem acoplar o Nº Mec. bem como o código do elemento de avaliação - HWA01, HWA02, FWSpat (Home Work Assigmente 01 e 02, Final Work for spatial part)

## Aspetos importantes a ter em conta na submissão do repositório para efeitos de avaliação 

A submissão do repositório para efeitos de avaliação deve ser feita através da plataforma moodle. A submissão deve ser feita até à data limite estabelecida no cronograma da disciplina. A avaliação dos trabalhos práticos será realizada com base nos outputs disponíveis nos "notebooks" jupyter ou, de preferência, em relatórios em formato pdf ou html gerados a partir dos mesmos. Devem garantir que os vossos ficheiros para avaliação têm todos os outputs necessários. Na avaliação não será executado código para gerar outputs em falta.

Ainda assim, para efeitos de reproducibilidade, aconselha-se a que substituam o ficheiro de ambiente de programação (yml) atualizado para o vosso projeto. Este aspeto é crucial no caso de procederem à instalação de bibliotecas adicionais. 

## Aspetos sobre segurança e partilha de informação 
A pasta 'data' e o ficheiro github .env não devem ser partilhados no repositório online GitHub!! Configurar corretamente o ficheiro '.gitignore' por forma assegurar que tal não ocorre. Alerta-se que a partilha online do ficheiro ".env" expõe toda a informação (privada) que possa aí estar registada. Alerta-se também que a versão gratuita do GitHub tem um espaço de armazenaento limitado (500MB) e que a partilha de ficheiros de dados pode exceder rapidamente esse limite. A pasta de dados deve ser partilhada com o docente por meios alteranativos (plataforma moodle, link onedrive) 


# Autor : [Paulo Batista](https://github.com/paulorlb])

 Março de 2026


# Sessions 3/4 Spatial Econometrics Notebook

This repository now includes a resumed and completed Sessions 3/4 lab notebook:

- `notebooks/Session_03_04_Spatial_Econometric_Models.qmd`
- backup before Prompt 6 resume: `notebooks/Session_03_04_Spatial_Econometric_Models_before_prompt6_resume.qmd`
- resume audit: `outputs/diagnostics/PROMPT_6_RESUME_AUDIT.md`

The notebook resumes from the staged Prompt 1-5 workflow and completes Prompt 6 material: SDM/SDEM, spatial impacts, robustness matrices, final synthesis, exercises, bibliography, and packaging.

## Required Files

The core notebook expects these files from the project root:

- `data/ETE_Lab.gpkg`
- `AGENTS.md`
- `NOTEBOOK_SPEC.md`
- `knowledge/KB_SOURCE_INVENTORY.md`
- `knowledge/KB_SESSION_03_04.md`
- `knowledge/KB_NOTEBOOK_MARKDOWN_PLAN.md`

Useful companion metadata:

- `data/dbPrimeYield_AVRILH_ETE_schema.md`
- `data/CAOP24_CONT_MUNI_metadata.md`
- `outputs/layer_inventory/ETE_Lab_layer_inventory.md`
- `outputs/layer_inventory/IMPLEMENTATION_PLAN.md`

## Expected Project Structure

```text
projETE/
  data/
    ETE_Lab.gpkg
    dbPrimeYield_AVRILH_ETE_schema.md
    CAOP24_CONT_MUNI_metadata.md
  knowledge/
    KB_SOURCE_INVENTORY.md
    KB_SESSION_03_04.md
    KB_NOTEBOOK_MARKDOWN_PLAN.md
  notebooks/
    Session_03_04_Spatial_Econometric_Models.qmd
  outputs/
    diagnostics/
    layer_inventory/
```

## Package Requirements

Install the packages listed in `requirements.txt`. Core packages include `numpy`, `pandas`, `geopandas`, `shapely`, `matplotlib`, `libpysal`, `esda`, `statsmodels`, and `spreg`.

Optional packages such as `mapclassify`, `splot`, `contextily`, and `osmnx` improve mapping or extension sections but are not required for the main Prompt 6 workflow.

## How To Run The Notebook

From the project root, start Jupyter or an editor with a Python kernel that has the required packages installed, then run:

```text
notebooks/Session_03_04_Spatial_Econometric_Models.qmd
```

The notebook inspects GeoPackage layers programmatically before assigning roles. It uses `data/ETE_Lab.gpkg` as the single authoritative spatial data source and uses EPSG:3763 for metric operations.

## How To Render Quarto Output

With Quarto installed and an appropriate Python/Jupyter kernel available:

```powershell
quarto render notebooks\Session_03_04_Spatial_Econometric_Models.qmd --to html
```

For a syntax/render dry-run without executing Python chunks:

```powershell
quarto render notebooks\Session_03_04_Spatial_Econometric_Models.qmd --to html --execute false
```

Quarto may still require a working Python/Jupyter bridge, including `jupyter` and `PyYAML`, even for no-execute rendering.

## Applied Examples

The notebook contains two applied tracks:

- **Aveiro/Ilhavo housing-market track:** starts from cleaned listing points, diagnoses coordinate and assignment quality, aggregates to zones and parishes/freguesias, builds Queen/Rook/kNN weights, estimates OLS and spatial models, computes SAR/SDM impacts, and tests robustness across supports, W definitions, outcomes, small-N restrictions, outlier flags, and fallback-share flags.
- **Mainland Portugal municipal track:** uses `CAOP24_CONT_MUNI` as the municipal support with housing-price targets and municipal indicators, diagnoses missingness and high-price municipalities, builds municipal weights, estimates parsimonious OLS/spatial models, computes impacts where feasible, and checks robustness across W definitions, targets, indicator families, extreme municipalities, and missingness.

## Known Limitations

- The data are observational asking-price listings and municipal indicators; the notebook avoids causal claims.
- Municipal housing-price targets are sparse and target-specific.
- Zone, parish/freguesia, and municipal results are support-dependent.
- Spatial weights are modelling assumptions, not facts.
- SAR and SDM coefficients are not marginal effects; the notebook computes direct, indirect, and total impacts through the spatial multiplier.
- Session 5 topics are intentionally kept out except for short forward-looking notes.
