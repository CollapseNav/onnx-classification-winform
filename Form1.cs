using System.Data;
using System.Diagnostics.CodeAnalysis;
using Microsoft.ML.OnnxRuntime;

namespace onnx_classification_winform;

public partial class Form1 : Form
{
    private InferenceSession? Inf;
    private const string ModelPath = "onnx_classification_winform.Model.mobilenetv2-10.onnx";
    private const string LabelPath = "onnx_classification_winform.Model.label_cn.txt";
    private string[] Labels = ModelLoader.LoadLabels(LabelPath);
    private Hardware hw;
    private ImageCut? ImageCut;
    private DataGridView Grid;
    private OriginPic OriginImage = new OriginPic();
    private OriginPic ResizeImage = new OriginPic();
    public Form1()
    {
        InitializeComponent();

        Button showLabels = new Button() { Text = "显示标签" };
        Controls.Add(showLabels);

        Button button = new Button() { Text = "选择图片", Location = new Point(showLabels.Location.X + showLabels.Width + 10, showLabels.Location.Y) };
        Controls.Add(button);

        InitImageCut(showLabels);
        InitRadioButton(button);
        InitGridView(button);

        button.Click += (_, _) =>
        {
            CheckImageForm();
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files (*.jpg;*.jpeg;*.png;*.gif)|*.jpg;*.jpeg;*.png;*.gif";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                var bitmap = new Bitmap(openFileDialog.FileName);
                GetResult(bitmap);

            }
        };

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
            Grid.DataSource = dt;
        };
    }

    /// <summary>
    /// 切换 InferenceSession
    /// </summary>
    /// <param name="hardware"></param>
    private void SwitchInf(Hardware hardware)
    {
        if (Inf != null)
            Inf.Dispose();
        Inf = ModelLoader.LoadModel(ModelPath, hardware);
    }

    /// <summary>
    /// 一个简单的Softmax计算
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    private IEnumerable<float> SoftMax(IEnumerable<float> output)
    {
        float sum = output.Sum(item => (float)Math.Exp(item));
        return output.Select(item => (float)Math.Exp(item) / sum);
    }

    /// <summary>
    /// 简单初始化单选按钮radio
    /// </summary>
    /// <param name="button"></param>
    public void InitRadioButton(Button button)
    {
        RadioButton useCpu = new RadioButton() { Text = "CPU", Checked = false };
        RadioButton useDml = new RadioButton() { Text = "DML", Checked = false };
        Controls.Add(useCpu);
        Controls.Add(useDml);
        useCpu.CheckedChanged += (_, _) => SwitchInf(Hardware.CPU);
        useDml.CheckedChanged += (_, _) => SwitchInf(Hardware.DML);
        useCpu.Checked = true;
        useCpu.Location = new Point(button.Location.X + button.Width + 10, button.Location.Y);
        useDml.Location = new Point(useCpu.Location.X + useCpu.Width, useCpu.Location.Y);
    }

    /// <summary>
    /// 简单初始化GridView
    /// </summary>
    /// <param name="button"></param>
    [MemberNotNull("Grid")]
    public void InitGridView(Button button)
    {
        Grid = new DataGridView();
        Controls.Add(Grid);
        Grid.AutoSize = true;
        Grid.Visible = false;
        Grid.Location = new Point(button.Location.X, button.Location.Y + button.Height + 10);
        button.Click += (_, _) => Grid.Visible = false;
        Grid.DataSourceChanged += (_, _) => Grid.Visible = true;
    }

    public void InitImageCut(Button button)
    {
        Button btn = new Button() { Text = "截图" };
        Controls.Add(btn);
        btn.Location = new Point(button.Location.X, button.Location.Y + button.Height + 10);

        btn.Click += (_, _) =>
        {
            OriginImage.Close();
            ResizeImage.Close();
            Hide();
            Thread.Sleep(300);
            var CurrentScreen = Screen.FromPoint(Cursor.Position);
            Bitmap bitmap = new(CurrentScreen.Bounds.Width, CurrentScreen.Bounds.Height);
            Graphics graphics = Graphics.FromImage(bitmap);
            graphics.CopyFromScreen(CurrentScreen.Bounds.Location, new Point(0, 0), CurrentScreen.Bounds.Size);
            //创建新的截图窗口，将刚才的屏幕作为背景
            ImageCut = new ImageCut { Owner = this, BackgroundImage = bitmap };
            ImageCut.Cut += GetResult;
            ImageCut.FormClosed += (_, _) => Show();
            ImageCut.ShowDialog();
        };
    }

    public void GetResult(Bitmap bitmap)
    {
        CheckImageForm();
        var inputName = Inf!.InputNames[0];
        var inputMetadata = Inf.InputMetadata[inputName];
        var dimensions = inputMetadata.Dimensions;
        dimensions[0] = 1;
        OriginImage.LoadImage(bitmap);
        OriginImage.Show();
        var resizeBitmap = bitmap.Resize(dimensions[2], dimensions[3]);
        ResizeImage.LoadImage(resizeBitmap);
        ResizeImage.Show();
        var inputs = resizeBitmap.Preprocess(Inf);
        using var results = Inf.Run(inputs);
        var outputs = SoftMax(results[0].AsEnumerable<float>());
        var result = outputs.Select((x, i) => new { x, label = Labels[i] })
                    .OrderByDescending(x => x.x)
                    .Take(6)
                    .ToArray();
        Grid.DataSource = result;
    }

    public void CheckImageForm()
    {
        if (OriginImage.IsDisposed)
            OriginImage = new OriginPic();
        if (ResizeImage.IsDisposed)
            ResizeImage = new OriginPic();
    }
}
