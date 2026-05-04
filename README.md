# Repositório do material de "live coding" da disciplina de ETE
Componente teoricó-prática da disciplina de [Econometria Temporal e Espacial](https://www.ua.pt/pt/uc/15124), do plano curricular do [Mestrado em Ciência dos Dados para Ciências Sociais](http://cdcs.web.ua.pt/?page_id=616) da [Universidade de Aveiro](https://www.ua.pt/pt/c/473/p).
Este repositório alberga os contéudos das sessões de código "ao vivo" da componente de econometria espacial. Pretende-se que este material permita consolidar os conceitos abordados  na aula, procurando  familiarizar os estudantes - na sua maioria das ciências sociais - com a utilização de métodos e técnicas de econometria espacial no contexto interdisciplinar do cientista de dados. 
A componente espacial desta disciplina assume que os alunos estão familiarizados com  a) análise estatística (inferência e modelos estatísticos de regressão) b) sistemas de informação geográfica, c) modelos de aprendizagem computacional.

Entre outros dados, este repositório recorre aos dados produzidos no âmbito do projeto [Avaliação de impacto territorial para operações de revitalização urbana](https://www.ua.pt/pt/projetos-id/1062) e da tese de doutoramento [A estrutura de interação de um sistema e-territorial: território, mercado de habitação e econometria espacial](https://ria.ua.pt/handle/10773/26275). Os dados disponibilizados encontram-se protegidos por direitos de autor  e a sua utilização fora do âmbito da disciplina das atividades desenvolvidas na disciplina de Econometria Temporal e Espacial da Universidade de Aveiro é **estritamente proibida** .

#### -- Estado do projeto: [ATIVO]




### Biblioteca indispensável
* [Pysal - Python Spatial Analysis Library](https://pysal.org/)
* BOOK:  [Geographic Data Science with Python](https://geographicdata.science/book/intro.html)


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


