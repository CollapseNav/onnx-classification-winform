public class ImageCut : Form
{
    //截图状态
    bool cutStart = false;
    //图像数据的byte数组
    public byte[] bytes;
    //矩形选取框范围
    Rectangle rectangle;
    //记录选取的起始点坐标
    Point downPoint;
    public Action<Bitmap> Cut;
    public ImageCut()
    {
        SuspendLayout();
        AutoScaleDimensions = new SizeF(6F, 12F);
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(Screen.PrimaryScreen?.Bounds.Width ?? 1920, Screen.PrimaryScreen?.Bounds.Height ?? 1080);
        FormBorderStyle = FormBorderStyle.None;
        ResumeLayout(false);
        KeyDown += ImageCut_KeyDown;
        MouseClick += ImageCut_MouseClick;
        MouseDown += ImageCut_MouseDown;
        MouseMove += ImageCut_MouseMove;
        MouseUp += ImageCut_MouseUp;
    }
    /// <summary>
    /// 按钮按下时触发
    /// Esc、Enter可以退出截图
    /// </summary>
    private void ImageCut_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Escape || e.KeyCode == Keys.Enter)
        {
            DialogResult = DialogResult.OK;
            Close();
        }
    }
    /// <summary>
    /// /// 鼠标右键可以退出截图
    /// </summary>
    private void ImageCut_MouseClick(object? sender, MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Right)
        {
            DialogResult = DialogResult.OK;
            Close();
            Owner?.Show();
        }
    }
    /// <summary>
    /// 鼠标左键按下时开始截图
    /// </summary>
    private void ImageCut_MouseDown(object? sender, MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Left)
        {
            //当前截图状态为false时，记录第一次按下时的坐标
            if (!cutStart)
            {
                cutStart = true;
                downPoint = new Point(e.X, e.Y);
            }
        }
    }
    /// <summary>
    /// 鼠标移动时画出选取的矩形框
    /// </summary>
    private void ImageCut_MouseMove(object? sender, MouseEventArgs e)
    {
        if (cutStart)
        {
            //新建一个背景的副本
            using Bitmap bitmap = new((Image)BackgroundImage!.Clone());
            //用于定位矩形的第二个点
            Point startPoint = new(downPoint.X, downPoint.Y);
            using Graphics graphics = Graphics.FromImage(bitmap);
            //框选一个矩形区域
            using Pen pen = new(Color.BurlyWood, 0.5f);
            int width = Math.Abs(e.X - downPoint.X);
            int height = Math.Abs(e.Y - downPoint.Y);

            startPoint.X = e.X < downPoint.X ? e.X : downPoint.X;
            startPoint.Y = e.Y < downPoint.Y ? e.Y : downPoint.Y;

            rectangle = new Rectangle(startPoint, new Size(width, height));
            //将矩形区域在副本上画出来画出来
            graphics.DrawRectangle(pen, rectangle);
            //将包含矩形区域的副本画出来
            using Graphics drawGraphics = this.CreateGraphics();
            drawGraphics.DrawImage(bitmap, new Point(0, 0));
        }
    }
    /// <summary>
    /// 松开鼠标
    /// </summary>
    private void ImageCut_MouseUp(object? sender, MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Left)
        {
            if (cutStart)
            {
                cutStart = false;
                if (rectangle.Width == 0 || rectangle.Height == 0)
                {
                    rectangle.Width = Screen.AllScreens[0].Bounds.Width;
                    rectangle.Height = Screen.AllScreens[0].Bounds.Height;
                    rectangle.Location = new Point(0, 0);
                }
                Bitmap bitmap = new(rectangle.Width, rectangle.Height);
                using Graphics graphics = Graphics.FromImage(bitmap);
                graphics.DrawImage((Image)BackgroundImage!.Clone(), new Rectangle(0, 0, rectangle.Width, rectangle.Height), rectangle, GraphicsUnit.Pixel);
                Close();
                Cut(bitmap);
            }
        }
    }

}