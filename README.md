# improved-diffusion

Este é um fork do repositório de código para [Improved Denoising Diffusion Probabilistic Models](https://github.com/openai/improved-diffusion).

# Uso

Este README foi adaptado do repositório original e explica como treinar e gerar amostras de um modelo.

## Instalação

Clone este repositório e navegue até ele no seu terminal. Em seguida, execute:

```
pip install -e .
```


Isso deve instalar o pacote Python `improved_diffusion` do qual os scripts dependem.

## Preparando os Dados

O código de treinamento lê imagens a partir de um diretório contendo arquivos de imagem.

Para criar seu próprio dataset, simplesmente coloque todas as suas imagens em um diretório com extensões ".jpg", ".jpeg" ou ".png". Se desejar treinar um modelo condicional por classe, nomeie os arquivos como "minhaetiqueta1_XXX.jpg", "minhaetiqueta2_YYY.jpg", etc., para que o carregador de dados saiba que "minhaetiqueta1" e "minhaetiqueta2" são as etiquetas. Subdiretórios também serão automaticamente enumerados, então as imagens podem estar organizadas de forma recursiva (embora os nomes dos diretórios sejam ignorados; os prefixos com underline são usados como nomes).

As imagens serão automaticamente redimensionadas e recortadas centralmente pelo pipeline de carregamento de dados. Basta passar `--data_dir caminho/para/imagens` para o script de treinamento, e ele cuidará do resto.

## Treinamento

Para treinar seu modelo, você deve primeiro decidir alguns hiperparâmetros. Aqui vai uma sugestão baseado no que usei para o treinamento do nosso modelo:

```
--attention_resolutions 32,16 --class_cond False --diffusion_steps 4000 --dropout 0.1 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_res_blocks 2 --use_fp16 True --use_scale_shift_norm True --lr 1e-4 --save_interval 20000 --batch_size 16
```
> **Observação**: O script original do improved diffusion não suporta treinamento para imagens 128x128, essa alteração no script original foi feita [neste commit](https://github.com/OPaivaHeitor/improved-diffusion/commit/b44f1b3edbc50ec7531140fefa81f3c25052a1ba)

Depois de configurar seus hiperparâmetros, você pode rodar um experimento assim:

```
python scripts/image_train.py --data_dir path/to/images --...
```
Ao executar o treinamento, os checkpoints serão salvos numa pasta que será criada dentro do diretório /tmp 

Você também pode querer treinar de forma distribuída. Neste caso, execute o mesmo comando com `mpiexec`:

```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```


Ao treinar de forma distribuída, você deve dividir manualmente o argumento `--batch_size` pelo número de processos (ranks). Em vez de usar treinamento distribuído, você pode usar `--microbatch 16` (ou `--microbatch 1` em casos extremos de limitação de memória) para reduzir o uso de memória.

Os logs e modelos salvos serão escritos em um diretório de log determinado pela variável de ambiente `OPENAI_LOGDIR`. Se ela não estiver definida, então um diretório temporário será criado em `/tmp`.

## Geração de Amostras

O script de treinamento acima salva checkpoints em arquivos `.pt` no diretório de logs. Esses checkpoints terão nomes como `ema_0.9999_200000.pt` e `model200000.pt`. Provavelmente você desejará amostrar a partir dos modelos EMA, já que produzem amostras muito melhores.

Depois de ter o caminho para seu modelo, você pode gerar um grande lote de amostras assim:

```
python scripts/image_sample.py --model_path /path/to/model.pt --hiperparâmetros
```


Novamente, isso salvará os resultados em um diretório de logs. As amostras são salvas como um grande arquivo `npz`, onde `arr_0` no arquivo é um grande lote de amostras.
