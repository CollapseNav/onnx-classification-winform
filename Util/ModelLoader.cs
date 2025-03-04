using Microsoft.ML.OnnxRuntime;

public class ModelLoader
{
    public static InferenceSession LoadModel(string modelPath, Hardware hardware = Hardware.CPU, int deviceid = 0)
    {
        SessionOptions options = new SessionOptions();
        if (hardware == Hardware.DML)
            options.AppendExecutionProvider_DML(deviceid);
        var inf = new InferenceSession(modelPath, options);
        return inf;
    }

    public static string[] LoadLabels(string labelPath)
    {
        return File.ReadAllText(labelPath).Trim().Split("\n");
    }
}