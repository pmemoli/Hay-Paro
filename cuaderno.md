# Cuaderno de Laboratorio

## Primer aproximacion (11/02/24)
Conseguimos el dataset clasificado de Clarin.

Entrenamos el modelo a partir de las noticias relacionadas a paro de Clarin. Con un validation set del 20% (60 noticias) obtenemos un accuracy del 76%. El batch_size es de 4, el lr de 0.001 y entrenamos 2 epochs (mas comienza a overfittear).

El resultado es bastante bueno y tardo muy poco en entrenar, que tiene sentido ya que partimos de un modelo BETO que ya es muy poderoso. De cualquier forma no tenemos buena nocion de como funciona el modelo para noticias no relacionadas a paro y de otras fuentes. Ademas el validation set es muy pequeño y el formato de noticias de Clarin posiblemente repita, esto no nos permite detectar overfitting.


## Obteniendo todas las noticias (16/02/24)
Terminamos de conseguir y clasificar el dataset para los tres medios (con chatgpt + revision nuestra).

## Decisiones de dataset y sampleo (19/02/24)

La cantidad de noticias no relacionadas a paros puede ser arbitrariamente grande, asi que decidimos que no excedan mas del doble de noticias de paro. Las proporciones terminan siendo: 

La division train-validation-test la decidimos en 60%, 20%, 20%. Asegurando que la proporcion de clases y medios sea la misma en los tres conjuntos.

### Bootstraping el dataset

Debatimos y pensamos 2 opciones para balancear el dataset y que el modelo aprenda
bien:

- Sobresamplear el dataset para tener un 60% de noticias relacionadas a paro 

- Bootstrapear con chatgpt 2000 noticias relacionadas a paro y 1000 que no.
  (Le mostramos a chatgpt un conjunto de 300 noticias en cada caso sampleadas sin reposicion y le pedimos
  que genere 2000 y 1000 respectivamente, distintas a las originales pero en el mismo estilo, e introduciendo
  pequeñas variaciones)
 
Comparamos estas proporciones con entrenar el modelo con el dataset original (proporcion 1:2 relacionadas
a paro vs no) usando el validation set.


## Pensando modelos a probar (22/02/24)
Adelantandonos a la posibilidad de que una unica FC layer sea muy simple y se quede con un accuracy
poco respetable, pensamos distintos modelos para experimentar. 

### Modelos base

Partimos de unos buenos embeddings contextuales (con BETO) y jugamos con modelos simples.     

- FC layers. Podemos probar con 1-2. 
- Bi-LSTMs. Podemos experimentar con el hidden state size, los stacks (1-2), etc.

### Modelos base con word2vec

Dado que es infactible finetunear incluso pedazitos de Bert con nuestras GPUs podemos
probar finetunear un modelo word2vec pre-entrenado a español con articulos, y usar esos 
embeddings en vez de los de Bert. 

No seria loco que el modelo base ya de suficientemente bien como para que esto no haga falta.

### Finetuneando Bert

Usamos un modelo similar a Bert muy destilado (12 millones de parametros) y probamos
finetunearle el ultimo bloque de atencion (1-3 millones de param) con una unica FC
layer para clasificar. La infernal cantidad de parametros seguro se aproveche bootstrapeando
muchisimos datos.


## Dataset completado (23/02/24)

Terminamos de scrapear las fuente que quedaba y dejamos todo bonito el pipeline de datos para 
experimentar. Separamos en test-validation-train en archivos distintos asegurando una estratificacion
por las clases de RELACIONADO A PARO y MEDIOS.  


## Experimentando modelos y tecnicas de resampleo (24/02/24)

Generamos 400 datos nuevos con chatgpt dandole el training set y pidiendole
que emule la distribucion original. Tambien le explicamos como clasificar
correctamente y lo logra.

Hicimos una experimentacion pre-eliminar con dos modelos para 3 datasets:
  2 Epochs 2fc model
  Basic: 0.77 precision positiva, 0.91 precision negativa, 0.88 accuracy 
  Oversample: 0.7 precision positiva, 0.96 precision negativa, 0.88 accuracy
  Chatgpt: 0.81 precision positiva, 0.89 precision negativa, 0.88 accuracy

  2 Epochs logistic regression model
  Basic: 0.73 precision positiva, 0.95 precision negativa, 0.89 accuracy 
  Oversample: 0.7 precision positiva, 0.96 precision negativa, 0.88 accuracy
  Chatgpt: 0.77 precision positiva, 0.89 precision negativa, 0.87 accuracy

A regresion logistica le cuesta bastante tener un buen precision. Tiende a 
clasificar 0 bastante. El modelo mas complejo captura mejor las sutilezas
de los titulos relacionados a paro.

Entrenar al modelo con chatgpt consistentemente le da mas precision para 
clases positivas, y sobresamplear no necesariamente genera el efecto
de mejor precision. 

La siguiente vez vamos a experimentar con los hiperparametros del 2fc model.


## Organizando codigo y nueva division de train-validation-test (25/02/24)

Como todo indica que vamos a tener los mejores resultados generando datos con chatgpt (y posiblemente intentemos llevarlo al extremo), va a ser muy importante tener mas datos en los sets de testeo para elegir hiperparametros. Como el dataset original es chico, consideramos que es importante aumentar la cantidad de datos para evaluar los modelos. Como estamos sobresampleando con chatgpt muy agresivamente, hacer cross validation es mucho mas complejo y sutil. Habria que generar un nuevo dataset por cada fold para evitar data leakage.

Pasamos a 60-20-20, regeneramos datos para balancear las clases relacionado-no relacionado, y organizamos el codigo para evaluar (validation set) muchos modelos en distintos datasets de forma rapida y escalable.

Experimentamos y obtenemos estos resultados:
{'logistic_reg':
  {'gpt': {'precision_pos': 0.78,
   'precision_neg': 0.99,
   'accuracy': 0.93},
  'oversample': {'precision_pos': 0.8, 'precision_neg': 0.9, 'accuracy': 0.88},
  'basic': {'precision_pos': 0.72, 'precision_neg': 0.95, 'accuracy': 0.88}},

 'simple_256':
  {'gpt': {'precision_pos': 0.76,
   'precision_neg': 0.96,
   'accuracy': 0.9},
  'oversample': {'precision_pos': 0.78,
   'precision_neg': 0.9,
   'accuracy': 0.87},
  'basic': {'precision_pos': 0.72, 'precision_neg': 0.97, 'accuracy': 0.89}},

 'simple_512':
 {'gpt': {'precision_pos': 0.8,
   'precision_neg': 0.95,
   'accuracy': 0.91},
  'oversample': {'precision_pos': 0.84,
   'precision_neg': 0.9,
   'accuracy': 0.89},
  'basic': {'precision_pos': 0.75, 'precision_neg': 0.94, 'accuracy': 0.89}}}

El dataset generado SIEMPRE da mejor accuracy que el resto y mas precision negativa. Aunque la precision positiva del dataset de oversample es marginalmente mas alta. Esto posiblemente se deba a que hay mayor proporcion de clases positivas en ese dataset (chatgpt genera una proporcion 1:1).

Con esto tiene sentido experimentar un poco mas con el dataset generado con una clase de paro mas balanceada, y ver si es totalmente superior al de oversample. Una vez hecho eso podemos dedicarnos totalmente a buscar un BUEN modelo y un dataset generado mas llevado al extremo (10k muestras??)


## Mejor balanceo de oversampleo y mas experimentación (26/02/24)

Antes el oversampleo era para balancear las clases de noticias relacionadas y no a paro. Otra posibilidad es balancear las noticias segun tanto su relacion a paro y su clasificacion como si confirmarn paro o no.

Experimentamos y nos quedamos finalmente con 

60% relacionadas a paro (40% pos, 20% neg), 40% no relacionadas

Para tanto el sobresampleo como la distribucion de datos generados por chatgpt


## Afinando la generacion de datos (27/04/24)

Notamos que si le pasamos el dataset y le pedimos a chatgpt que genere datos asi nomas, cae en repetir unos cuantos titulos con pequeñas variaciones que no se asemejan mucho a la distribucion original:

"Huelga de trabajadores del sector de la venta online por condiciones de trabajo y derechos laborales",1

"Levantamiento de paro en la industria de la energía geotérmica tras acuerdos sobre desarrollo y explotación sostenible",0

"Convocan a huelga en el sector de la animación por reconocimiento del trabajo creativo y condiciones justas",1

"Desconvocan paro en el sector de la atención al cliente por mejoras en las condiciones laborales y formación",0

"Anuncian paro en la industria de la construcción por seguridad laboral y respeto a los derechos de los trabajadores",1

"Cancelan huelga en el sector de la educación física por reconocimiento y mejora de las instalaciones deportivas",0

"Huelga de trabajadores de la industria alimentaria por transparencia, calidad de los productos y condiciones laborales",1

"Levantamiento de paro en el sector de la asistencia sanitaria tras acuerdo sobre recursos y mejoras en el sistema de salud",0

"Convocan a huelga en la industria del transporte fluvial por condiciones de trabajo y mantenimiento de las vías navegables",1

"Desconvocan paro en el sector de la producción de energía solar por acuerdos sobre inversión y desarrollo de proyectos",0

"Anuncian paro en la industria del libro por derechos de autor, distribución justa y apoyo a las librerías independientes",1

"Cancelan huelga en el sector de la hostelería por medidas de apoyo al empleo y condiciones laborales tras la crisis sanitaria",0

"Huelga de trabajadores del sector de la moda por sostenibilidad, condiciones éticas de producción y transparencia",1

Encontramos una muy buena forma de evitar esto que es explicarle bien el problema a chatgpt, y pedirle que genere datos en tandas de 50. Para cada tanda le mostramos 5 o 6 títulos del train dataset y le pedimos que se inspire en esos títulos.
Generar de a tandas chicas nos permite evaluar a ojo la calidad del dataset.

Asi logramos que emule bastante bien la distribución original pero manteniendo bastante variación y creatividad. Ademas no cae en repetir frases con algunas palabras cambiadas.


# Evaluando los modelos (08/03/24)

(Esto es el resultado de unos cuantos días...)

Terminamos de entrenar los modelos. Elegimos estos dos:

- 2 fc layers de 768 * seq length y 256 parámetros. Batch size de 128
- Bi-LSTM de 768 x, 128 h y c. Batch size de 128

Los modelos entrenados con el dataset de sobresampleo presenta el mejor rendimiento
y parecen entrenarse en su punto justo con menos epochs. Sorprendentemente, 
los modelos entrenados con los datasets de chatgpt y el básico terminan dando
aproximadamente igual.

Para el dataset de sobresampleo la lstm da considerablemente mejor que un modelo simple de dos layers, mientras que para GPT y BASIC dan lo mismo.


# Testeando con bi-lstm oversample (20/03/24)

El dataset de testeo tiene casi 500 datos (28% clase positiva), obtuvimos los siguientes resultados (redondeado a dos decimas):

- precision: 0.82
- recall 0.84
- accuracy: 0.92
- accuracy relacionados: 0.75
- accuracy no-relacionados: 0.99
