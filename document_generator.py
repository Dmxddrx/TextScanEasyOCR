
import pymupdf

from PIL import Image


def sort_results(results):

    return sorted(
        results,
        key=lambda item: (
            min(point[1] for point in item["bbox"]),
            min(point[0] for point in item["bbox"])
        )
    )


def generate_searchable_pdf(pages, output_path):

    pdf = pymupdf.open()

    for page_number, page_data in enumerate(
        pages,
        start=1
    ):

        image_path = page_data["image"]

        results = sort_results(
            page_data["results"]
        )

        # Read image dimensions
        with Image.open(image_path) as image:

            image_width, image_height = image.size

        # Maintain the image aspect ratio
        page_width = 595.276

        page_height = (
            page_width * image_height / image_width
        )

        page = pdf.new_page(
            width=page_width,
            height=page_height
        )

        # Insert original colour page
        page.insert_image(
            page.rect,
            filename=str(image_path)
        )

        scale_x = page_width / image_width
        scale_y = page_height / image_height

        # Insert invisible text
        for item in results:

            text = item["text"].strip()

            if not text:
                continue

            bbox = item["bbox"]

            x0 = min(p[0] for p in bbox) * scale_x
            y0 = min(p[1] for p in bbox) * scale_y

            x1 = max(p[0] for p in bbox) * scale_x
            y1 = max(p[1] for p in bbox) * scale_y

            rect = pymupdf.Rect(
                max(0, x0 - 0.5),
                max(0, y0 - 0.5),
                min(page_width, x1 + 1.5),
                min(page_height, y1 + 1.5)
            )

            box_height = rect.height

            inserted = False

            # Retry with smaller font if the text does not fit
            for factor in (0.70, 0.55, 0.40, 0.30):

                fontsize = max(
                    2.5,
                    box_height * factor
                )

                spare = page.insert_textbox(
                    rect,
                    text,
                    fontsize=fontsize,
                    fontname="helv",
                    render_mode=3,
                    overlay=True
                )

                if spare >= 0:

                    inserted = True
                    break

            if not inserted:

                # Fallback: retain searchable text
                page.insert_text(
                    (rect.x0, min(page_height - 2, rect.y1)),
                    text,
                    fontsize=3,
                    fontname="helv",
                    render_mode=3
                )

        print(
            f"PDF page completed: {page_number}"
        )

    pdf.save(str(output_path))

    pdf.close()

    print("\nSearchable PDF created:")
    print(output_path)


def generate_txt(pages, output_path):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for page_number, page_data in enumerate(
            pages,
            start=1
        ):

            file.write(
                f"\n\n========== PAGE {page_number} ==========\n\n"
            )

            results = sort_results(
                page_data["results"]
            )

            for item in results:

                text = item["text"].strip()

                if text:

                    file.write(text + "\n")

    print("\nTXT file created:")
    print(output_path)
