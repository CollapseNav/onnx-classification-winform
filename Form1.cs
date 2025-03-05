using System.Data;
using Microsoft.ML.OnnxRuntime;

namespace onnx_classification_winform;

public partial class Form1 : Form
{
    private InferenceSession? Inf;
    private const string ModelPath = "onnx_classification_winform.Model.mobilenetv2-10.onnx";
    private const string LabelPath = "onnx_classification_winform.Model.label_cn.txt";
    private string[] Labels = ModelLoader.LoadLabels(LabelPath);
    public Form1()
    {
        InitializeComponent();

        Button button = new Button();
        Controls.Add(button);
        button.Text = "选择图片";

        // 添加radiobtn
        RadioButton useCpu = new RadioButton();
        RadioButton useDml = new RadioButton();
        Controls.Add(useCpu);
        Controls.Add(useDml);
        useCpu.Text = "使用CPU";
        useDml.Text = "使用DML";
        useCpu.Location = new Point(button.Location.X + button.Width + 10, button.Location.Y);
        useDml.Location = new Point(useCpu.Location.X + useCpu.Width, useCpu.Location.Y);
        useCpu.Checked = true;

        var grid = new DataGridView();
        Controls.Add(grid);
        grid.AutoSize = true;
        grid.Visible = false;
        grid.Location = new Point(button.Location.X, button.Location.Y + button.Height + 10);

        OriginPic form = new OriginPic();
        OriginPic form2 = new OriginPic();
        button.Click += (_, _) =>
        {
            grid.Visible = false;
            if (useCpu.Checked)
                SwitchInf(Hardware.CPU);
            else if (useDml.Checked)
                SwitchInf(Hardware.DML);

            var inputName = Inf!.InputNames[0];
            var inputMetadata = Inf.InputMetadata[inputName];
            var dimensions = inputMetadata.Dimensions;
            dimensions[0] = 1;

            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files (*.jpg;*.jpeg;*.png;*.gif)|*.jpg;*.jpeg;*.png;*.gif";
            if (form.IsDisposed)
                form = new OriginPic();
            if (form2.IsDisposed)
                form2 = new OriginPic();
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
                grid.Visible = true;
                grid.Refresh();
            }
        };

        Button showLabels = new Button();
        Controls.Add(showLabels);
        showLabels.Text = "显示标签";
        showLabels.Location = new Point(useDml.Location.X + useDml.Width + 10, useDml.Location.Y);

        showLabels.Click += (_, _) =>
        {
            DataTable dt = new DataTable();
            dt.Columns.AddRange(Enumerable.Range(0, 10).Select(item => new DataColumn(item.ToString())).ToArray());
            Labels.Chunk(10)
            .Select(labels => labels.Select((label, index) => new { label, index })
            .ToDictionary(item => item.index, item => item.label)).ToList()
            .ForEach(item =>
            {
                var row = dt.NewRow();
                row.ItemArray = item.Values.ToArray();
                dt.Rows.Add(row);
            });
            grid.DataSource = dt;
            grid.Visible = true;
            grid.ScrollBars = ScrollBars.Vertical;
            grid.Location = new Point(button.Location.X, button.Location.Y + button.Height + 10);
        };
    }

    private void SwitchInf(Hardware hardware)
    {
        if (Inf != null)
            Inf.Dispose();
        Inf = ModelLoader.LoadModel(ModelPath, hardware);
    }

    private IEnumerable<float> SoftMax(IEnumerable<float> output)
    {
        float sum = output.Sum(item => (float)Math.Exp(item));
        return output.Select(item => (float)Math.Exp(item) / sum);
    }
}
