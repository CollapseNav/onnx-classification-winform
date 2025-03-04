using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public static class BitmapExt
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] StdDev = [0.229f, 0.224f, 0.225f];
    public static Bitmap Resize(this Bitmap bitmap, int width, int height)
    {
        Bitmap resizedBitmap = new(width, height);
        using (Graphics graphics = Graphics.FromImage(resizedBitmap))
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(bitmap, 0, 0, width, height);
        }
        return resizedBitmap;
    }

    public static List<NamedOnnxValue> Preprocess(this Bitmap bitmap, InferenceSession inf)
    {
        var inputName = inf.InputNames[0];
        var inputMetadata = inf.InputMetadata[inputName];
        var dimensions = inputMetadata.Dimensions;
        dimensions[0] = 1;
        var tensor = new DenseTensor<float>(dimensions);
        BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

        int stride = bmpData.Stride;
        IntPtr ptr = bmpData.Scan0;
        int bytes = Math.Abs(stride) * bitmap.Height;
        byte[] rgbValues = new byte[bytes];

        Marshal.Copy(ptr, rgbValues, 0, bytes);

        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                int index = y * stride + x * 3;
                byte blue = rgbValues[index];
                byte green = rgbValues[index + 1];
                byte red = rgbValues[index + 2];

                tensor[0, 0, y, x] = ((red / 255f) - Mean[0]) / StdDev[0];
                tensor[0, 1, y, x] = ((green / 255f) - Mean[1]) / StdDev[1];
                tensor[0, 2, y, x] = ((blue / 255f) - Mean[2]) / StdDev[2];
            }
        }

        bitmap.UnlockBits(bmpData);

        return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
    }
}