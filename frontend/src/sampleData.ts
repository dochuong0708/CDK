export interface SampleArticle {
  id: string
  title: string
  category: string
  text: string
}

export const SAMPLE_ARTICLES: SampleArticle[] = [
  {
    id: 'sample-1',
    title: 'Cảnh báo khẩn cấp: Khóa tài khoản định danh VNeID trong 24h',
    category: 'Lừa đảo / Phishing',
    text: `CHIA SẺ GẤP CHO NGƯỜI THÂN! CẢNH BÁO KHẨN CẤP!

Bộ Công an vừa phát đi thông báo chấn động: Tất cả tài khoản định danh VNeID mức 2 chưa cập nhật sinh trắc học sẽ bị khóa tài khoản vĩnh viễn trong vòng 24 giờ tới. Người dân không thực hiện ngay sẽ bị phạt nguội và khởi tố theo quy định mới.

Để tránh bị khóa thẻ ngân hàng và phạt tiền, bà con hãy truy cập ngay vào đường link bit.ly/xac-minh-vneid-2024 để cập nhật lại thông tin cá nhân, cung cấp mã OTP và stk ngân hàng nhằm đồng bộ dữ liệu.

Một cán bộ giấu tên cho biết nếu không xác minh ngay hôm nay, toàn bộ số tiền trong tài khoản sẽ bị đóng băng. Hãy chia sẻ ngay kẻo muộn cho bạn bè và người thân cùng cảnh giác!`,
  },
  {
    id: 'sample-2',
    title: 'Chấn động: Phát hiện thần dược lá đu đủ chữa dứt điểm ung thư',
    category: 'Y tế / Giật gân',
    text: `TIẾT LỘ ĐỘNG TRỜI MÀ CÁC TẬP ĐOÀN DƯỢC PHẨM CỐ TÌNH CHE GIẤU!

Tin đồn trên mạng xã hội suốt thời gian qua đã được chứng minh là có thật. Một chuyên gia giấu tên vừa công bố bài thuốc thần dược từ lá đu đủ và sả có khả năng chữa dứt điểm ung thư giai đoạn cuối, cam kết 100% không tái phát.

Rất nhiều bệnh nhân bệnh viện trả về đã khỏi hẳn chỉ sau 3 tuần uống nước sắc lá này. Tuy nhiên, sự thật rợn người là các bệnh viện lớn và giới y khoa quốc tế đang bưng bít thông tin này vì sợ mất doanh thu bán thuốc hóa trị hàng tỷ đồng.

Bà con đừng vội tin vào bác sĩ Tây y, hãy chia sẻ ngay kẻo muộn bài thuốc cứu mạng này đến các nhóm kín và người quen để tự cứu lấy mình!`,
  },
  {
    id: 'sample-3',
    title: 'Hà Nội tăng cường kiểm tra an toàn thực phẩm dịp cuối năm',
    category: 'Chính thống / Tin chuẩn',
    text: `UBND thành phố Hà Nội vừa ban hành kế hoạch cao điểm về kiểm tra, kiểm soát thị trường và bảo đảm an toàn vệ sinh thực phẩm dịp cuối năm và Tết Nguyên đán.

Theo đó, các đoàn kiểm tra liên ngành từ cấp thành phố đến quận, huyện sẽ tập trung thanh tra các cơ sở sản xuất, chế biến, kinh doanh thực phẩm tươi sống, bánh kẹo và đồ uống có cồn. Cơ quan chức năng yêu cầu xử lý nghiêm các trường hợp vi phạm về nhãn mác, nguồn gốc xuất xứ và điều kiện bảo quản.

Sở Y tế Hà Nội khuyến cáo người tiêu dùng nên chọn mua thực phẩm tại các hệ thống siêu thị, cửa hàng có giấy phép chứng nhận an toàn, không mua sản phẩm trôi nổi không rõ nguồn gốc trên thị trường.`,
  },
]

export interface SampleMediaItem {
  id: string
  title: string
  category: string
  fileName: string
  fileSize: string
  expectedRisk: number
  label: string
  description: string
  previewType: string
}

export const SAMPLE_IMAGES: SampleMediaItem[] = [
  {
    id: 'img-sample-1',
    title: 'Giấy triệu tập Công an đóng dấu đỏ giả mạo',
    category: 'Văn bản giả mạo',
    fileName: 'giay_trieu_tap_cong_an_gia.jpg',
    fileSize: '412 KB',
    expectedRisk: 88,
    label: 'Cực kỳ đáng ngờ (Review recommended)',
    description: 'Con dấu có dấu hiệu ghép đè (overlay), sai lệch font chữ chuẩn của Bộ Công an, viền chữ ký bị nhòe pixel (compression artifact).',
    previewType: 'document',
  },
  {
    id: 'img-sample-2',
    title: 'Ảnh khuôn mặt tạo bởi AI (Deepfake / Midjourney)',
    category: 'Ảnh chân dung AI',
    fileName: 'ai_portrait_synthetic.png',
    fileSize: '1.2 MB',
    expectedRisk: 74,
    label: 'Khả năng cao do AI tạo (Synthetic Face)',
    description: 'Phát hiện bất đối xứng ở vành tai và đồng tử mắt, vùng tóc hòa trộn bất thường với phông nền (blur artifact), thiếu dữ liệu EXIF máy ảnh.',
    previewType: 'face',
  },
  {
    id: 'img-sample-3',
    title: 'Ảnh chụp hiện trường vụ việc thực tế',
    category: 'Ảnh thực tế (An toàn)',
    fileName: 'hien_truong_chup_that.jpg',
    fileSize: '2.4 MB',
    expectedRisk: 18,
    label: 'Độ tin cậy cao (Likely authentic)',
    description: 'Đầy đủ siêu dữ liệu máy ảnh Sony Alpha, mức nhiễu cảm biến (sensor noise) đồng nhất trên toàn bộ khung hình.',
    previewType: 'scene',
  },
]

export const SAMPLE_AUDIOS: SampleMediaItem[] = [
  {
    id: 'audio-sample-1',
    title: 'Cuộc gọi Deepfake Voice Clone mượn tiền khẩn cấp',
    category: 'Giả giọng AI (Voice Clone)',
    fileName: 'cuoc_goi_gia_giong_muon_tien.mp3',
    fileSize: '840 KB',
    expectedRisk: 84,
    label: 'Khả năng cao là giọng AI (Synthetic Voice)',
    description: 'Phát hiện tần số âm sắc không tự nhiên (spectral discontinuity), không có nhịp lấy hơi sinh học, độ ngắt quãng máy móc lặp lại.',
    previewType: 'voice_scam',
  },
  {
    id: 'audio-sample-2',
    title: 'Bản ghi âm giọng nói đe dọa đòi nợ giả danh tòa án',
    category: 'Đe dọa / Tự động (Robocall)',
    fileName: 'thong_bao_toa_an_robocall.wav',
    fileSize: '1.6 MB',
    expectedRisk: 76,
    label: 'Cuộc gọi tự động / Đe dọa (Review recommended)',
    description: 'Tốc độ phát âm cố định (robotic cadence), dải tần bị cắt gọt nhân tạo dưới 4kHz, văn phong đe dọa khóa tài khoản.',
    previewType: 'robocall',
  },
  {
    id: 'audio-sample-3',
    title: 'Bản tin phát thanh thời sự VOV',
    category: 'Âm thanh thực tế (An toàn)',
    fileName: 'ban_tin_thoi_su_vov.mp3',
    fileSize: '3.1 MB',
    expectedRisk: 12,
    label: 'Bản ghi thật (Likely authentic)',
    description: 'Dải âm thanh tự nhiên, có tiếng nền môi trường và độ rung thanh quản chuẩn của giọng đọc con người.',
    previewType: 'natural',
  },
]

