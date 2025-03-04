using Microsoft.ML.OnnxRuntime;

namespace onnx_classification_winform;

public partial class Form1 : Form
{
    private InferenceSession Inf;
    private string[] Labels;
    public Form1()
    {
        InitializeComponent();
        Labels = ModelLoader.LoadLabels("onnx_classification_winform.Model.label_cn.txt");
        Inf = ModelLoader.LoadModel("onnx_classification_winform.Model.mobilenetv2-10.onnx", Hardware.DML);
        var inputName = Inf.InputNames[0];
        var inputMetadata = Inf.InputMetadata[inputName];
        var dimensions = inputMetadata.Dimensions;
        dimensions[0] = 1;
        Button button = new Button();
        Controls.Add(button);
        button.Text = "选择图片";
        var grid = new DataGridView();
        Controls.Add(grid);
        grid.AutoSize = true;
        grid.Visible = false;
        OriginPic form = new OriginPic();
        OriginPic form2 = new OriginPic();
        button.Click += (_, _) =>
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files (*.jpg;*.jpeg;*.png;*.gif)|*.jpg;*.jpeg;*.png;*.gif";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var bitmap = new Bitmap(openFileDialog.FileName);
                form.LoadImage(bitmap);
                form.Show();
                var resizeBitmap = bitmap.Resize(dimensions[2], dimensions[3]);
                form2.LoadImage(resizeBitmap);
                form2.Show();
                var inputs = resizeBitmap.Preprocess(Inf);
                using var results = Inf.Run(inputs);
                var outputs = SoftMax(results[0].AsEnumerable<float>());
                var result = outputs.Select((x, i) => new { x, label = Labels[i] })
                            .OrderByDescending(x => x.x)
                            .Take(6)
                            .ToArray();
                grid.DataSource = result;
                grid.Location = new Point(button.Location.X, button.Location.Y + button.Height + 10);
                grid.Visible = true;
            }
        };

    }

    private IEnumerable<float> SoftMax(IEnumerable<float> output)
    {
        float sum = output.Sum(item => (float)Math.Exp(item));
        return output.Select(item => (float)Math.Exp(item) / sum);
    }
}
