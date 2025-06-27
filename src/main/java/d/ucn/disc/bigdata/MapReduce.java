package d.ucn.disc.bigdata;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Framework MapReduce personalizado usando Apache Spark como motor
 * Implementa la estructura del profesor pero con procesamiento distribuido real
 */
public final class MapReduce {

    // Clase KeyValue genérica proporcionada por el profesor
    public static class KeyValue<K, V> implements Serializable {
        private static final long serialVersionUID = 1L;
        private final K key;
        private final V value;

        public KeyValue(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() { return key; }
        public V getValue() { return value; }

        @Override
        public String toString() {
            return "(" + key + ", " + value + ")";
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            KeyValue<?, ?> that = (KeyValue<?, ?>) obj;
            return Objects.equals(key, that.key) && Objects.equals(value, that.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(key, value);
        }
    }

    // Interface Mapper proporcionada por el profesor
    public interface Mapper<T, K, V> extends Serializable {
        List<KeyValue<K, V>> map(T item);
    }

    // Interface Reducer proporcionada por el profesor
    public interface Reducer<K, V, R> extends Serializable {
        KeyValue<K, R> reduce(K key, List<V> values);
    }

    // Clase para representar una línea con su número
    public static class LineWithNumber implements Serializable {
        private static final long serialVersionUID = 1L;
        public final String content;
        public final int lineNumber;

        public LineWithNumber(String content, int lineNumber) {
            this.content = content;
            this.lineNumber = lineNumber;
        }

        @Override
        public String toString() {
            return "Línea " + lineNumber + ": " + content;
        }
    }

    /**
     * Método principal MapReduce usando Apache Spark como motor distribuido
     * numWorkers se usa para configurar el paralelismo de Spark
     */
    public static <T, K, V, R> Map<K, R> mapReduce(
            List<T> inputData,
            Mapper<T, K, V> mapper,
            Reducer<K, V, R> reducer,
            int numWorkers) {

        // Configurar Spark
        SparkConf conf = new SparkConf()
                .setAppName("MapReduceFramework")
                .setMaster("local[" + (numWorkers <= 0 ? "*" : numWorkers) + "]")
                .set("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
                .set("spark.driver.host", "localhost")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");

        JavaSparkContext sparkContext = new JavaSparkContext(conf);
        sparkContext.setLogLevel("WARN");

        try {
            // FASE MAP: Usar Spark RDD para procesamiento distribuido
            JavaRDD<T> inputRDD = sparkContext.parallelize(inputData,
                    numWorkers <= 0 ? sparkContext.defaultParallelism() : numWorkers);

            // Aplicar la función mapper a cada elemento
            JavaRDD<KeyValue<K, V>> mappedRDD = inputRDD.flatMap(item -> {
                return mapper.map(item).iterator();
            });

            // Convertir a PairRDD para operaciones de agrupamiento
            JavaPairRDD<K, V> pairRDD = mappedRDD.mapToPair(kv ->
                    new Tuple2<>(kv.getKey(), kv.getValue())
            );

            // FASE SHUFFLE: Agrupar por clave usando Spark
            JavaPairRDD<K, Iterable<V>> groupedRDD = pairRDD.groupByKey();

            // FASE REDUCE: Aplicar reducer a cada grupo
            JavaPairRDD<K, R> reducedRDD = groupedRDD.mapValues(values -> {
                List<V> valueList = new ArrayList<>();
                values.forEach(valueList::add);

                K key = null; // Se obtiene del contexto de Spark
                return reducer.reduce(key, valueList).getValue();
            });

            // Ajuste para obtener la clave correcta en el reduce
            JavaRDD<KeyValue<K, R>> finalRDD = groupedRDD.map(tuple -> {
                K key = tuple._1();
                Iterable<V> values = tuple._2();

                List<V> valueList = new ArrayList<>();
                values.forEach(valueList::add);

                return reducer.reduce(key, valueList);
            });

            // Recopilar resultados
            List<KeyValue<K, R>> results = finalRDD.collect();

            // Convertir a Map
            Map<K, R> finalResults = new HashMap<>();
            for (KeyValue<K, R> result : results) {
                finalResults.put(result.getKey(), result.getValue());
            }

            return finalResults;

        } finally {
            sparkContext.close();
        }
    }

    /**
     * Implementación específica del Mapper para Índice Invertido
     */
    public static class InvertedIndexMapper implements Mapper<LineWithNumber, String, Integer> {
        private static final long serialVersionUID = 1L;
        private static final Pattern WORD_PATTERN = Pattern.compile("[\\p{Punct}\\s]+");

        @Override
        public List<KeyValue<String, Integer>> map(LineWithNumber line) {
            List<KeyValue<String, Integer>> results = new ArrayList<>();

            // Tokenizar la línea en palabras
            String[] words = WORD_PATTERN.split(line.content.toLowerCase().trim());

            for (String word : words) {
                word = word.trim();
                // Filtrar palabras válidas
                if (!word.isEmpty() && word.length() > 1 && word.matches("[a-zA-Z]+")) {
                    results.add(new KeyValue<>(word, line.lineNumber));
                }
            }

            return results;
        }
    }

    /**
     * Implementación específica del Reducer para Índice Invertido
     */
    public static class InvertedIndexReducer implements Reducer<String, Integer, Set<Integer>> {
        private static final long serialVersionUID = 1L;

        @Override
        public KeyValue<String, Set<Integer>> reduce(String word, List<Integer> lineNumbers) {
            // Crear conjunto único de números de línea
            Set<Integer> uniqueLines = new TreeSet<>(lineNumbers);
            return new KeyValue<>(word, uniqueLines);
        }
    }

    /**
     * Clase principal para demostrar el Índice Invertido con Spark
     */
    public static class InvertedIndexDemo {

        /**
         * Muestra información del archivo procesado
         */
        public static void showFileInfo(String filename, Map<String, Set<Integer>> index) {
            File file = new File(filename);
            if (file.exists()) {
                long fileSize = file.length();
                System.out.println("\n=== INFORMACIÓN DEL ARCHIVO ===");
                System.out.println("Nombre: " + filename);
                System.out.println("Tamaño: " + fileSize + " bytes");
                System.out.println("Palabras únicas encontradas: " + index.size());

                int totalOccurrences = index.values().stream().mapToInt(Set::size).sum();
                System.out.println("Total de ocurrencias de palabras: " + totalOccurrences);
                System.out.println("Procesado con Apache Spark (distribuido)");
            }
        }

        /**
         * Crea el índice invertido usando el framework MapReduce con Spark
         */
        public static Map<String, Set<Integer>> createInvertedIndex(String filename, int numWorkers) {
            List<LineWithNumber> inputData = readFile(filename);

            if (inputData.isEmpty()) {
                return new HashMap<>();
            }

            System.out.println("Iniciando procesamiento distribuido con Apache Spark...");
            System.out.println("Workers/Particiones: " + (numWorkers <= 0 ? "Auto (todos los cores)" : numWorkers));

            InvertedIndexMapper mapper = new InvertedIndexMapper();
            InvertedIndexReducer reducer = new InvertedIndexReducer();

            // Usar Apache Spark para el procesamiento distribuido
            return mapReduce(inputData, mapper, reducer, numWorkers);
        }
    /*
        /**
         * Muestra las variables creadas para cada palabra
         */

        public static void displayWordVariables(Map<String, Set<Integer>> index) {
            System.out.println("\n=== VARIABLES CREADAS POR MAPREDUCE (APACHE SPARK) ===");

            List<String> sortedWords = new ArrayList<>(index.keySet());
            Collections.sort(sortedWords);

            // Mostrar solo las primeras 20 para archivos grandes
            for (String word : sortedWords) {
                Set<Integer> lines =index.get(word);
                System.out.println("Palabra: " +word + " en la/s Lineas: " + lines);
            }


        }

        /**
         * Analiza y muestra las palabras más encontradas
         */
        public static void analyzeWordFrequency(Map<String, Set<Integer>> index) {
            System.out.println("\n=== ANÁLISIS DE PALABRAS MÁS ENCONTRADAS (SPARK) ===");

            // Crear lista de palabras ordenadas por frecuencia
            List<Map.Entry<String, Set<Integer>>> wordsByFrequency = new ArrayList<>(index.entrySet());
            wordsByFrequency.sort((a, b) -> Integer.compare(b.getValue().size(), a.getValue().size()));

            System.out.println("Top 15 palabras más encontradas:");
            for (int i = 0; i < Math.min(15, wordsByFrequency.size()); i++) {
                Map.Entry<String, Set<Integer>> entry = wordsByFrequency.get(i);
                String word = entry.getKey();
                int frequency = entry.getValue().size();
                Set<Integer> lines = entry.getValue();

                // Para palabras muy frecuentes, mostrar solo algunos números de línea
                String lineDisplay = lines.size() > 10 ?
                        "[" + lines.stream().limit(5).map(String::valueOf).reduce((a,b) -> a + ", " + b).orElse("") + ", ...]" :
                        lines.toString();

                System.out.println((i + 1) + ". " + word + " - Aparece en " + frequency +
                        " línea(s): " + lineDisplay);
            }

            // Estadísticas generales
            int totalWords = index.size();
            int totalOccurrences = index.values().stream().mapToInt(Set::size).sum();
            double avgOccurrences = (double) totalOccurrences / totalWords;

            System.out.println("\nEstadísticas del procesamiento distribuido:");
            System.out.println("- Total de palabras únicas: " + totalWords);
            System.out.println("- Total de ocurrencias: " + totalOccurrences);
            System.out.println("- Promedio de ocurrencias por palabra: " + String.format("%.2f", avgOccurrences));
            System.out.println("- Procesado usando Apache Spark RDDs distribuidos");
        }

        /**
         * Función que permite leer un archivo y contar su cantidad de línea
         * @param filename archivo que se lee en formato .txt
         * @return int de cantidad de lineas.
         */
        private static List<LineWithNumber> readFile(String filename) {
            List<LineWithNumber> lines = new ArrayList<>();

            try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
                String line;
                int lineNumber = 1;

                while ((line = reader.readLine()) != null) {
                    // Solo agregar líneas que no estén vacías
                    if (!line.trim().isEmpty()) {
                        lines.add(new LineWithNumber(line, lineNumber));
                    }
                    lineNumber++;
                }

                System.out.println("Archivo leído exitosamente: " + lines.size() + " líneas con contenido");

            } catch (FileNotFoundException e) {
                System.err.println("Error: Archivo no encontrado - " + filename);
            } catch (IOException e) {
                System.err.println("Error leyendo archivo: " + e.getMessage());
            } catch (Exception e) {
                System.err.println("Error inesperado: " + e.getMessage());
            }

            return lines;
        }

    }

    /**
     * Método main para ejecutar la demostración con Apache Spark
     */
    public static void main(String[] args) {
        String filename;

        // Permitir especificar archivo por argumentos de línea de comandos
        if (args.length > 0) {
            filename = args[0];
            System.out.println("Usando archivo especificado: " + filename);
        } else {
            // Archivo por defecto - cambia este nombre por tu archivo personalizado
            filename = "shakespeare.txt";
        }

        // Verificar si el archivo existe, si no, crear archivo de ejemplo
        File file = new File(filename);
        if (!file.exists()) {
            System.out.println("Archivo '" + filename + "' no encontrado.");

            // Solo crear archivo de ejemplo si es el archivo por defecto
            if (args.length == 0) {
                System.err.println("Error: El archivo especificado no existe.");
                return;
            }
        } else {
            System.out.println("Usando archivo existente: " + filename);
        }

        // Ejecutar MapReduce con Apache Spark
        System.out.println("Ejecutando MapReduce con Apache Spark...");
        Map<String, Set<Integer>> invertedIndex = InvertedIndexDemo.createInvertedIndex(filename, 0);

        if (invertedIndex.isEmpty()) {
            System.err.println("No se pudo procesar el archivo. Verifica que existe y contiene texto.");
            return;
        }

        // Mostrar información del archivo
        InvertedIndexDemo.showFileInfo(filename, invertedIndex);

        // Mostrar las variables creadas por MapReduce
        InvertedIndexDemo.displayWordVariables(invertedIndex);

        // Analizar cuáles fueron las palabras más encontradas
        InvertedIndexDemo.analyzeWordFrequency(invertedIndex);

        System.out.println("\n=== PROCESAMIENTO COMPLETADO CON APACHE SPARK ===");
    }
}