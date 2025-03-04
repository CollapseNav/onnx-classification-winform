using System.Reflection;
using System.Text;
using Microsoft.ML.OnnxRuntime;

public class ModelLoader
{
    public static InferenceSession LoadModel(string modelPath, Hardware hardware = Hardware.CPU, int deviceid = 0)
    {
        Assembly assembly = Assembly.GetExecutingAssembly();
        using Stream stream = assembly.GetManifestResourceStream(modelPath);
        byte[] bytes = new byte[stream.Length];
        stream.Read(bytes, 0, bytes.Length);
        SessionOptions options = new SessionOptions();
        if (hardware == Hardware.DML)
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.AppendExecutionProvider_DML(deviceid);
        }
        var inf = new InferenceSession(bytes, options);
        return inf;
    }

    public static string[] LoadLabels(string labelPath)
    {
        Assembly assembly = Assembly.GetExecutingAssembly();
        using Stream stream = assembly.GetManifestResourceStream(labelPath);

        byte[] bytes = new byte[stream.Length];
        stream.Read(bytes, 0, bytes.Length);
        return Encoding.UTF8.GetString(bytes).Trim().Split("\n");
    }
}